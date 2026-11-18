"""
Arabic Dialect Detector Engine
Specialized for Saudi, Bahraini, Lebanese, and MSA (Modern Standard Arabic).
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
You are an elite Arabic sociolinguist and dialectologist specializing in phonetics, morphology, and dialect identification.
You will listen directly to an audio excerpt from a YouTube video or podcast.

Your task is to identify which Arabic dialect is spoken.

THE 4 TARGET CLASSES ARE:
1. Saudi (including Najdi, Hejazi, Sharqawi, and Southern accents)
2. Bahraini (including Bahrani and Sunni/Gulf Bahraini accents)
3. Lebanese (the most lebanese dialects spoken in Lebanon, including urban and rural variations)
4. MSA (Modern Standard Arabic / الفصحى used in formal media, news, and speeches)

IF THE AUDIO IS IN ANY OTHER DIALECT (e.g. Egyptian, Moroccan/Darija, Algerian, Tunisian, Syrian, Palestinian, Iraqi, Sudanese) OR NON-ARABIC:
-> Classify as: "Other" (and state the detected language/dialect in parentheses, e.g. "Other (Egyptian)").

CRITICAL EVALUATION RULES:
1. FOCUS STRICTLY ON THE ARABIC BASE:
   - Disregard foreign code-switching or loanwords (e.g., English tech words or French words like 'bonjour', 'merci', 'cava'). Judge solely based on the Arabic phonology, grammar, lexical terms, and intonation.
2. LONG-FORM & MULTI-SPEAKER / MIXED REGISTERS:
   - Provide the dominant dialect overall.
   - Provide an estimated percentage distribution across ALL 4 target classes plus Other. The percentages MUST sum to 100%.
3. LINGUISTIC EVIDENCE:
   - Identify specific key vocabulary/phrases heard in the audio that guided your decision (e.g., 'هلا والله', 'وشو', 'عشان', 'دحين', 'شخبارك', 'وايد', 'صج', 'شو هيدا', 'هيك', 'كتير', 'عم', etc.).
   - Note phonological traits (e.g., realization of Qaf [ق] as [g], [ʔ/hamza], or [q]; Jeem [ج] as [y] or [ʒ]; Kaf [ك] as [ch/ts]; vowel elongation and imala).

OUTPUT FORMAT:
You must output your final response structured exactly with these headers:

Dominant Dialect: <Saudi | Bahraini | Lebanese | MSA | Other>
Confidence: <percentage, e.g. 92%>

Distribution:
- Saudi: <X>%
- Bahraini: <Y>%
- Lebanese: <Z>%
- MSA: <W>%
- Other: <K>%

Linguistic Evidence:
- Distinctive Vocabulary: <list specific words/idioms heard>
- Phonological Markers: <realization of consonants/vowels, accents>
- Intonation & Cadence: <rhythm, pitch patterns, sentence melody>
- Dialect Nuance & Notes: <observations on sub-dialect or code-switching if applicable>
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
        """
        dominant = "Unknown"
        confidence = "N/A"
        distribution = {"Saudi": 0.0, "Bahraini": 0.0, "Lebanese": 0.0, "MSA": 0.0, "Other": 0.0}

        # Match Dominant Dialect
        dom_match = re.search(r"Dominant Dialect:\s*([^\n\r]+)", full_text, re.IGNORECASE)
        if dom_match:
            raw_dom = dom_match.group(1).strip()
            # Normalize class
            for target in ["Saudi", "Bahraini", "Lebanese", "MSA", "Other"]:
                if target.lower() in raw_dom.lower():
                    dominant = target
                    break
            if dominant == "Unknown":
                dominant = raw_dom

        # Match Confidence
        conf_match = re.search(r"Confidence:\s*(\d+%)", full_text, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).strip()

        # Match Distribution percentages
        for key in ["Saudi", "Bahraini", "Lebanese", "MSA", "Other"]:
            pat = rf"-\s*{key}:\s*(\d+(?:\.\d+)?)\s*%"
            m = re.search(pat, full_text, re.IGNORECASE)
            if m:
                distribution[key] = float(m.group(1))

        # Check if percentages sum to roughly 100; if not normalized, normalize
        total = sum(distribution.values())
        if total > 0:
            distribution_normalized = {k: v / total for k, v in distribution.items()}
        else:
            distribution_normalized = {k: 0.0 for k in distribution}

        return {
            "dominant_dialect": dominant,
            "confidence": confidence,
            "distribution_raw": distribution,
            "distribution_normalized": distribution_normalized,
            "full_analysis": full_text
        }
