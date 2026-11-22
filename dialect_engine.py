"""
Arabic Dialect Detector Engine
Open dialect identification with multi-speaker detection and percentage breakdowns.
Powered by the latest working Google Gemini model and in-memory audio extraction.
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

# Default Gemini model: "auto" dynamically selects the latest working Gemini model
DEFAULT_MODEL = os.getenv("GEMINI_DIALECT_MODEL", "auto")

SYSTEM_PROMPT = """
You are an elite Arabic sociolinguist and dialectologist specializing in Arabic phonetics, morphology, acoustic cues, and dialect identification.
You will listen directly to an audio excerpt from a video, podcast, or recording.

Your task is to identify the Arabic dialect spoken by EVERY speaker in the audio excerpt.
DO NOT restrict yourself to any predefined list of dialects. Identify any Arabic dialect or regional sub-dialect present (e.g., Egyptian [Cairene, Alexandrian, Sa'idi], Levantine [Lebanese, Syrian, Palestinian, Jordanian], Gulf / Khaleeji [Saudi - Najdi/Hejazi/Sharqawi/Southern, Emirati, Kuwaiti, Bahraini, Qatari, Omani], Iraqi [Baghdadi, Southern, Moslawi], Yemeni, Sudanese, Maghrebi [Moroccan Darija, Algerian, Tunisian, Libyan], Modern Standard Arabic / MSA [الفصحى], etc.).

CRITICAL EVALUATION RULES:
Analyze the Arabic audio and identify each speaker separately.

1. First identify how many distinct speakers are present.

2. Assign a unique identifier to each speaker:
   Speaker 1, Speaker 2, Speaker 3, etc.

3. Analyze each speaker's dialect INDEPENDENTLY.
   Do not assume speakers share the same dialect.

4. Use each speaker's own pronunciation, vocabulary,
   and grammatical patterns as evidence.

5. Identify the most specific dialect supported by
   the audio (e.g., Emirati, Lebanese, Saudi, Egyptian).

6. Do not label a speaker simply as "Gulf" or
   "Levantine" when sufficient evidence supports
   a more specific dialect.

7. If there is not enough evidence for a specific
   dialect, use the broader dialect family and
   mark the prediction as uncertain.

8. If one speaker dominates the conversation,
   do not automatically assign their dialect
   to the other speakers.

9. Provide a short explanation of the linguistic
   evidence for each speaker.

10. Do not invent evidence or claim certainty
    when the audio is ambiguous.

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
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
            )
        self.client = genai.Client(api_key=self.api_key)

        # Resolve primary model: if "auto", "latest", or None, dynamically select the latest working model
        target = (model_name or os.getenv("GEMINI_DIALECT_MODEL") or "auto").strip()
        if target.lower() in ("auto", "latest", "auto (latest working model)", ""):
            discovered = self.discover_latest_gemini_models()
            if discovered:
                self.model_name = discovered[0]
                logger.info(f"Dynamically resolved latest working primary model: {self.model_name}")
            else:
                self.model_name = "gemini-2.5-flash"
                logger.info(f"Using default primary model: {self.model_name}")
        else:
            self.model_name = target

        self.active_model = self.model_name

    @staticmethod
    def parse_time_offset(val: Any) -> int:
        """
        Parses flexible start time inputs into seconds.
        Supports:
          - Numbers / Floats: 2700 -> 2700
          - Timestamp colons: '45:00' -> 2700, '01:15:30' -> 4530, '00:30' -> 30
          - Human units: '45m' / '45min' -> 2700, '1h 30m' -> 5400, '30s' -> 30
          - Plain numbers: '35' -> 35 mins (2100s), '2700' -> 2700s
        """
        if val is None:
            return 0
        if isinstance(val, (int, float)):
            return max(0, int(val))

        val_str = str(val).strip()
        if not val_str:
            return 0

        # 1. Colon timestamps: MM:SS or HH:MM:SS
        if ":" in val_str:
            parts = [p.strip() for p in val_str.split(":")]
            try:
                if len(parts) == 2:
                    return max(0, int(parts[0]) * 60 + int(float(parts[1])))
                elif len(parts) >= 3:
                    return max(0, int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2])))
            except ValueError:
                pass

        # 2. String with units: e.g. 1h 30m, 45m, 30s
        total_sec = 0
        matched = False

        h_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hour|hours)", val_str, re.IGNORECASE)
        if h_m:
            total_sec += int(float(h_m.group(1)) * 3600)
            matched = True

        m_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|minute|minutes)(?!s)", val_str, re.IGNORECASE)
        if m_m:
            total_sec += int(float(m_m.group(1)) * 60)
            matched = True

        s_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|second|seconds)", val_str, re.IGNORECASE)
        if s_m:
            total_sec += int(float(s_m.group(1)))
            matched = True

        if matched:
            return max(0, total_sec)

        # 3. Plain numeric string (e.g. '35', '45', '1.5', '2700')
        try:
            num = float(val_str)
            # If >= 360, it was already given in raw seconds (e.g. 1800, 2700s)
            # If < 360, treat as minutes (e.g. 35 -> 35 min = 2100s, 45 -> 45 min = 2700s)
            if num >= 360:
                return max(0, int(num))
            else:
                return max(0, int(num * 60))
        except ValueError:
            return 0

    @staticmethod
    def format_timestamp(seconds: int) -> str:
        """Formats integer seconds into clean MM:SS or H:MM:SS format."""
        seconds = max(0, int(seconds))
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        secs = seconds % 60
        if hrs > 0:
            return f"{hrs}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def extract_audio_from_youtube(
        self,
        url: str,
        start_seconds: Any = 60,
        duration_seconds: int = 90
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Extracts an audio slice directly from a YouTube URL in-memory using yt-dlp and ffmpeg pipe.
        Zero disk usage for high speed and minimal bandwidth.
        """
        start_seconds = self.parse_time_offset(start_seconds)

        # Check if URL itself has a timestamp e.g. &t=45m or ?t=2700
        url_ts = re.search(r"[?&]t=([0-9hms]+)", url)
        if url_ts and start_seconds in (0, 60):
            parsed_url_ts = self.parse_time_offset(url_ts.group(1))
            if parsed_url_ts > 0:
                start_seconds = parsed_url_ts
                logger.info(f"Auto-detected timestamp in YouTube URL: {url_ts.group(1)} -> {start_seconds}s")
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
        start_seconds: Any = 0,
        duration_seconds: int = 90
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Extracts an in-memory audio slice from a locally uploaded audio or video file.
        Serves as a robust fallback if YouTube restricts access.
        """
        start_seconds = self.parse_time_offset(start_seconds)
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

    def discover_latest_gemini_models(self, exclude: Optional[list] = None) -> list:
        """
        Dynamically queries the Gemini API for available models and filters for
        the latest working generative multimodal models, sorted by recency and capabilities.
        No hardcoded fallback models are used.
        """
        exclude_set = set(exclude or [])
        candidate_models: list = []

        try:
            logger.info("Querying Gemini API for latest available working models...")
            api_models = self.client.models.list()

            scored_models = []
            for m in api_models:
                m_name = getattr(m, "name", "")
                if not m_name:
                    continue
                clean_name = m_name.replace("models/", "").replace("publishers/google/models/", "")

                if clean_name in exclude_set or m_name in exclude_set:
                    continue

                name_lower = clean_name.lower()
                if "gemini" not in name_lower:
                    continue

                # Filter out specialized, non-chat, or embedding models
                if any(bad in name_lower for bad in ["embedding", "aqa", "imagen", "veo", "tts", "whisper", "adapter", "realtime", "live"]):
                    continue

                # Verify supported actions if available
                actions = getattr(m, "supported_actions", None)
                if actions and "generateContent" not in actions:
                    continue

                is_latest = 1 if "latest" in name_lower else 0
                v_match = re.search(r"gemini[^\d]*(\d+(?:\.\d+)?)", name_lower)
                version = float(v_match.group(1)) if v_match else (99.0 if is_latest else 1.0)
                is_flash = 1 if "flash" in name_lower else 0
                is_stable = 0 if any(tag in name_lower for tag in ["preview", "experimental", "exp"]) else 1

                scored_models.append((version, is_flash, is_stable, clean_name))

            # Sort by: Version (highest/latest first), then Flash (best for fast audio streaming), then Stable
            scored_models.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            candidate_models = [item[3] for item in scored_models]
            logger.info(f"Dynamically discovered working Gemini models: {candidate_models}")

        except Exception as err:
            logger.error(f"Failed to query model list from Gemini API: {err}")

        return candidate_models

    def stream_dialect_identification(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/mp3",
        custom_instructions: str = ""
    ) -> Generator[str, None, None]:
        """
        Streams token chunks from Gemini as it listens to the audio and performs linguistic analysis.
        If the primary model fails, dynamically queries Gemini for the latest working model.
        """
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        user_content = SYSTEM_PROMPT
        if custom_instructions:
            user_content += f"\n\nAdditional User Context:\n{custom_instructions}"

        primary_model = self.model_name
        logger.info(f"Sending audio slice to primary Gemini model: {primary_model}")

        # 1. Attempt with primary requested model
        try:
            stream = self.client.models.generate_content_stream(
                model=primary_model,
                contents=[audio_part, user_content]
            )
            has_yielded = False
            for chunk in stream:
                if chunk.text:
                    has_yielded = True
                    yield chunk.text
            if has_yielded:
                self.active_model = primary_model
                return
        except Exception as primary_err:
            logger.warning(
                f"Primary model '{primary_model}' failed ({primary_err}). "
                "Checking Gemini API for the latest available working models..."
            )

        # 2. Dynamic Fallback: Query Gemini API directly for latest working models
        discovered_models = self.discover_latest_gemini_models(exclude=[primary_model])
        if not discovered_models:
            raise RuntimeError(
                f"Primary model '{primary_model}' failed, and no working fallback model could be discovered from Gemini API."
            )

        last_error = None
        for fallback_model in discovered_models:
            try:
                logger.info(f"Attempting fallback with latest working Gemini model: {fallback_model}")
                stream = self.client.models.generate_content_stream(
                    model=fallback_model,
                    contents=[audio_part, user_content]
                )
                has_yielded = False
                for chunk in stream:
                    if chunk.text:
                        has_yielded = True
                        yield chunk.text
                if has_yielded:
                    self.active_model = fallback_model
                    logger.info(f"Successfully generated analysis using fallback model '{fallback_model}'")
                    return
            except Exception as fb_err:
                logger.warning(f"Fallback model '{fallback_model}' failed ({fb_err}). Checking next available model...")
                last_error = fb_err

        raise RuntimeError(f"All available Gemini models failed. Last error: {last_error}")

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
