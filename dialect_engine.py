"""
Arabic Dialect Detector Engine
Open dialect identification with multi-speaker detection and percentage breakdowns.
Powered by Google Gemini 3.1 Flash and in-memory audio extraction.
"""

import os
import re
import io
import json
import logging
import subprocess
from typing import Generator, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import imageio_ffmpeg
import static_ffmpeg
import yt_dlp
from google import genai
from google.genai import types

load_dotenv()

# Ensure both ffmpeg and ffprobe are in PATH for Gradio and subprocesses
static_ffmpeg.add_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
FFMPEG_DIR = os.path.dirname(FFMPEG_PATH)
if FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

# Default Gemini model
DEFAULT_MODEL = os.getenv("GEMINI_DIALECT_MODEL", "gemini-3.1-flash-lite")
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]

SYSTEM_PROMPT = """
You are an elite Arabic sociolinguist and dialectologist specializing in Arabic phonetics, morphology, acoustic cues, and dialect identification.
You will listen directly to an audio excerpt from a video, podcast, or recording.

Your task is to identify the Arabic dialect spoken by EVERY speaker in the audio excerpt.
DO NOT restrict yourself to any predefined list of dialects. Identify any Arabic dialect or regional sub-dialect present (e.g., Egyptian [Cairene, Alexandrian, Sa'idi], Levantine [Lebanese, Syrian, Palestinian, Jordanian], Gulf / Khaleeji [Saudi - Najdi/Hejazi/Sharqawi/Southern, Emirati, Kuwaiti, Bahraini, Qatari, Omani], Iraqi [Baghdadi, Southern, Moslawi], Yemeni, Sudanese, Maghrebi [Moroccan Darija, Algerian, Tunisian, Libyan], Modern Standard Arabic / MSA [الفصحى], etc.).

CRITICAL EVALUATION RULES:
1. MANDATORY FULL-PANEL SPEAKER DIARIZATION (3, 4, 5+ SPEAKERS):
   - DO NOT assume there are only two speakers (Host and Guest).
   - If there are 3, 4, 5, or more distinct individuals speaking (such as in roundtable discussions, talk shows, debates, or multi-host podcasts), you MUST track and list EVERY SINGLE UNIQUE SPEAKER separately.
   - Distinguish speakers by vocal pitch, timbre, gender, turn-taking, and conversational dynamics.
   - For EVERY speaker detected (Speaker 1, Speaker 2, Speaker 3, Speaker 4, etc.), identify their dialect and provide their estimated speaking percentage share of the excerpt.
   - If only 1 speaker is speaking, output Speaker 1 (100%).
2. OPEN DIALECT BREAKDOWN:
   - Provide a percentage breakdown of all dialects heard in the excerpt. The percentages across all detected dialects must sum to 100%.
   - Do NOT limit to any fixed categories; dynamically name the exact dialect(s) detected.
3. FOCUS STRICTLY ON THE ARABIC BASE:
   - Disregard foreign code-switching or loanwords (e.g., English tech words or French words like 'bonjour', 'merci', 'cava'). Judge solely based on Arabic phonology, grammar, lexical roots, and intonation.
4. LINGUISTIC EVIDENCE:
   - List key distinctive vocabulary and expressions heard from each speaker (e.g., 'دحين', 'كدة', 'عشان', 'شكو ماكو', 'وايد', 'صج', 'هيك', 'كتير', 'عم', 'بزاف', etc.).
   - Note phonological markers (e.g., realization of Qaf [ق] as [g], [ʔ/hamza], [q]; Jeem [ج] as [y], [ʒ], or [g]; Kaf [ك] as [ch/ts]; vowel lengthening, imala).
   - Note intonation, tempo, and pitch patterns for each speaker.

OUTPUT FORMAT:
You must output your response structured exactly with these headers:

Dominant Dialect: <Main dialect spoken overall, e.g. Egyptian / Najdi Saudi / Lebanese / etc.>
Confidence: <percentage, e.g. 92%>

Speakers Breakdown:
- Speaker 1: <Dialect Name> (<Percentage>%) - <Brief role or vocal trait>
- Speaker 2: <Dialect Name> (<Percentage>%) - <Brief role or vocal trait>
- Speaker 3: <Dialect Name> (<Percentage>%) - <Brief role or vocal trait>
- Speaker 4: <Dialect Name> (<Percentage>%) - <Brief role or vocal trait>
(CRITICAL: Enumerate ALL distinct speakers heard: Speaker 1, 2, 3, 4, 5... Do NOT stop at 2 speakers if more voices speak!)

Dialect Distribution:
- <Dialect 1>: <Percentage>%
- <Dialect 2>: <Percentage>%
(List all detected dialects with their percentage of the excerpt, summing to 100%)

Linguistic Evidence:
- Distinctive Vocabulary: <words/expressions heard, attributed to speakers>
- Phonological Markers: <consonant and vowel pronunciation details for each speaker>
- Intonation & Cadence: <rhythm, pitch patterns, sentence melody>
- Regional Nuance & Observations: <sub-dialect notes, urban vs. rural, or register shift>
"""

class DialectDetector:
    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def extract_audio_from_youtube(
        self,
        url: str,
        start_seconds: int = 60,
        duration_seconds: int = 90
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Extracts an audio slice directly from a YouTube URL in-memory using yt-dlp and ffmpeg pipe.
        Zero disk usage for high speed and minimal bandwidth.
        """
        logger.info(f"Extracting video info from URL: {url}")
        ydl_opts = {
            "format": "bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as e:
            logger.error(f"Failed to fetch YouTube info: {e}")
            raise RuntimeError(f"Could not retrieve YouTube video info: {str(e)}")

        video_title = info.get("title", "Unknown Title")
        video_duration = info.get("duration", 0)
        stream_url = info.get("url")

        if not stream_url:
            raise RuntimeError("Could not find a playable direct audio stream for this YouTube video.")

        # Adjust start time if video is shorter than requested start
        if video_duration and start_seconds >= video_duration:
            start_seconds = max(0, int(video_duration / 4))
            logger.info(f"Adjusted start_seconds to {start_seconds}s because video length is {video_duration}s")

        logger.info(f"Slicing audio stream: start={start_seconds}s, duration={duration_seconds}s")

        user_agent = info.get("http_headers", {}).get(
            "User-Agent",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-user_agent", user_agent,
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-ss", str(start_seconds),
            "-i", stream_url,
            "-t", str(duration_seconds),
            "-vn",
            "-ac", "1",            # Mono is optimal for speech
            "-ar", "16000",        # 16kHz speech recognition standard
            "-b:a", "64k",         # Lightweight MP3
            "-f", "mp3",
            "pipe:1"
        ]

        audio_bytes = None
        try:
            proc = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                check=True,
                timeout=45
            )
            audio_bytes = proc.stdout
        except Exception as pipe_err:
            logger.warning(f"Direct stream pipe failed ({pipe_err}). Falling back to yt-dlp partial download...")
            # Fallback: Use yt-dlp native range download into a temporary directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="dialect_dl_")
            out_tmpl = os.path.join(temp_dir, "slice.%(ext)s")
            dl_opts = {
                "format": "bestaudio/best",
                "outtmpl": out_tmpl,
                "download_ranges": yt_dlp.utils.download_range_func(None, [(start_seconds, start_seconds + duration_seconds)]),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "64",
                }],
                "quiet": True,
                "no_warnings": True,
                "overwrites": True,
            }
            try:
                with yt_dlp.YoutubeDL(dl_opts) as ydl:
                    ydl.download([url])
                target_file = os.path.join(temp_dir, "slice.mp3")
                if os.path.exists(target_file):
                    with open(target_file, "rb") as f:
                        audio_bytes = f.read()
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as dl_err:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError(f"Audio extraction failed on both stream pipe and range download: {dl_err}")

        if not audio_bytes or len(audio_bytes) < 1000:
            raise RuntimeError("Extracted audio stream is empty or too short.")

        metadata = {
            "title": video_title,
            "duration": video_duration,
            "channel": info.get("uploader", "Unknown Channel"),
            "start_time": start_seconds,
            "slice_duration": duration_seconds,
            "audio_size_bytes": len(audio_bytes),
            "thumbnail": info.get("thumbnail", "")
        }

        logger.info(f"Successfully extracted {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.1f} KB)")
        return audio_bytes, metadata

    def extract_audio_from_file(
        self,
        file_path: str,
        start_seconds: int = 0,
        duration_seconds: int = 90
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Extracts an in-memory audio slice from a locally uploaded audio or video file.
        Serves as a robust fallback if YouTube restricts access.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing local file fallback: {file_path}")
        ffmpeg_cmd = [
            FFMPEG_PATH,
            "-ss", str(start_seconds),
            "-i", file_path,
            "-t", str(duration_seconds),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            "-f", "mp3",
            "pipe:1"
        ]

        proc = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        audio_bytes = proc.stdout

        metadata = {
            "title": os.path.basename(file_path),
            "duration": None,
            "channel": "Local Upload",
            "start_time": start_seconds,
            "slice_duration": duration_seconds,
            "audio_size_bytes": len(audio_bytes),
            "thumbnail": ""
        }
        return audio_bytes, metadata

    def stream_dialect_identification(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/mp3",
        custom_instructions: str = ""
    ) -> Generator[str, None, None]:
        """
        Streams token chunks from Gemini Flash as it listens to the audio and performs linguistic analysis.
        """
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        user_content = SYSTEM_PROMPT
        if custom_instructions:
            user_content += f"\n\nAdditional User Context:\n{custom_instructions}"

        models_to_try = [self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]
        
        last_error = None
        for model in models_to_try:
            try:
                logger.info(f"Sending audio slice to Gemini model: {model}")
                stream = self.client.models.generate_content_stream(
                    model=model,
                    contents=[audio_part, user_content]
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:
                logger.warning(f"Model {model} failed with: {e}. Trying fallback if available...")
                last_error = e

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    @staticmethod
    def parse_analysis_result(full_text: str) -> Dict[str, Any]:
        """
        Parses the structured response text into high-level metrics for UI cards and charts.
        Dynamically extracts open dialects and per-speaker breakdowns without fixed list restrictions.
        """
        dominant = "Analyzing..."
        confidence = "N/A"
        speakers = []
        speaker_dist: Dict[str, float] = {}
        dialect_dist: Dict[str, float] = {}

        # Match Dominant / Primary Dialect
        dom_match = re.search(r"(?:Dominant|Primary)\s*Dialect:\s*([^\n\r]+)", full_text, re.IGNORECASE)
        if dom_match:
            dominant = dom_match.group(1).strip("* ")

        # Match Confidence
        conf_match = re.search(r"Confidence:\s*(\d+%)", full_text, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).strip()

        # Parse Speakers Breakdown (supports 3, 4, 5+ speakers, markdown headers, and variants)
        spk_section_pattern = (
            r"^\s*(?:#+\s*)?(?:Speakers?|Diarization)(?!\s*\d)(?:\s+(?:Breakdown|Diarization|&\s*Dialects?|List|Analysis|Overview|[^\n:]*))?\s*:\s*"
            r"([\s\S]*?)(?=^\s*(?:#+\s*)?(?:(?:Dialect\s+)?Distribution|Linguistic Evidence|Evidence)\s*:|\Z)"
        )
        spk_section_match = re.search(spk_section_pattern, full_text, re.MULTILINE | re.IGNORECASE)
        target_spk_text = spk_section_match.group(1) if spk_section_match else full_text

        lines = target_spk_text.strip().split("\n")
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Extract percentage (e.g., (30%), [25%], - 20%, : 15%)
            pct_m = re.search(
                r"(?:\([^\d%]*(\d+(?:\.\d+)?)\s*%\s*[^\)]*\)|\[[^\d%]*(\d+(?:\.\d+)?)\s*%\s*\]|\s*-\s*(\d+(?:\.\d+)?)\s*%|:\s*(\d+(?:\.\d+)?)\s*%)",
                line_clean
            )
            if not pct_m:
                continue
            pct = float([g for g in pct_m.groups() if g is not None][0])

            sub_line = re.sub(r"^\s*[\*\-]\s*", "", line_clean).strip()
            colon_idx = sub_line.find(":")
            if colon_idx == -1:
                continue

            spk_label = sub_line[:colon_idx].strip("* ")
            rest = sub_line[colon_idx + 1:].strip()

            # Ensure this is a speaker identifier
            is_spk = bool(re.search(r"(?:Speaker\s*\d+|Host|Guest|Panelist|Participant|Caller|Voice|Narrator|المتحدث)", spk_label, re.IGNORECASE))
            if not is_spk and not spk_section_match:
                continue

            # Handle e.g. Speaker 1 (Egyptian): 60%
            if "(" in spk_label and ")" in spk_label:
                m_spk = re.match(r"^(.*?)\s*\((.*?)\)$", spk_label)
                if m_spk:
                    speaker_name = m_spk.group(1).strip("* ")
                    dialect_name = m_spk.group(2).strip("* ")
                    note = rest
                    speakers.append({
                        "speaker": speaker_name,
                        "dialect": dialect_name,
                        "percentage": pct,
                        "note": note
                    })
                    speaker_dist[f"{speaker_name} ({dialect_name})"] = pct
                    continue

            speaker_name = spk_label
            pct_str = pct_m.group(0)
            pct_pos = rest.find(pct_str)
            dialect_name = rest[:pct_pos].strip(" -:*()") if pct_pos != -1 else rest
            note = rest[pct_pos + len(pct_str):].strip(" -:*") if pct_pos != -1 else ""
            speakers.append({
                "speaker": speaker_name,
                "dialect": dialect_name,
                "percentage": pct,
                "note": note
            })
            speaker_dist[f"{speaker_name} ({dialect_name})"] = pct

        # Parse Dialect Distribution section dynamically
        dist_section_match = re.search(
            r"^\s*(?:#+\s*)?(?:Dialect\s+)?Distribution\s*:\s*([\s\S]*?)(?=^\s*(?:#+\s*)?(?:Linguistic Evidence|Evidence|Speakers? Breakdown)\s*:|\Z)",
            full_text,
            re.MULTILINE | re.IGNORECASE
        )
        if dist_section_match:
            lines = dist_section_match.group(1).strip().split("\n")
            for line in lines:
                m = re.match(r"^\s*[\*\-]?\s*([^:\n\r]+?):\s*(\d+(?:\.\d+)?)\s*%", line)
                if m:
                    dialect_dist[m.group(1).strip("* ")] = float(m.group(2))

        # Determine best breakdown for charts
        if speaker_dist:
            chart_dist = speaker_dist
        elif dialect_dist:
            chart_dist = dialect_dist
        else:
            chart_dist = {}

        total = sum(chart_dist.values())
        if total > 0:
            distribution_normalized = {k: v / total for k, v in chart_dist.items()}
        else:
            distribution_normalized = {k: 0.0 for k in chart_dist}

        return {
            "dominant_dialect": dominant,
            "confidence": confidence,
            "speakers": speakers,
            "speaker_distribution": speaker_dist,
            "dialect_distribution": dialect_dist,
            "distribution_raw": chart_dist,
            "distribution_normalized": distribution_normalized,
            "full_analysis": full_text
        }
