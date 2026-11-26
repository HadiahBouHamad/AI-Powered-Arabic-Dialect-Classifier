"""
Unit Tests for DialectDetector.extract_audio_from_youtube
Covers:
  1. Successful direct stream extraction via FFmpeg pipe
  2. Timestamp parsing (string timestamps e.g. '01:30', '45m')
  3. URL timestamp auto-detection (e.g. '&t=45m')
  4. Video duration adjustment when requested start exceeds duration
  5. Error handling when YouTube extraction fails (invalid URL / network error)
  6. Error handling when stream URL is missing
  7. Fallback to yt-dlp partial download when direct pipe fails
  8. Error handling when both direct pipe and fallback fail
  9. Error handling when extracted audio is empty or too short (< 1000 bytes)
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import subprocess

from dialect_engine import DialectDetector


class TestExtractAudioFromYouTube(unittest.TestCase):
    def setUp(self):
        # Instantiate DialectDetector with a mocked Client and explicit model to avoid network calls
        with patch("google.genai.Client"):
            self.detector = DialectDetector(api_key="test-api-key", model_name="gemini-test")

        self.sample_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        self.sample_info = {
            "title": "Arabic Podcast Episode 1",
            "duration": 600,
            "url": "https://googlevideo.com/videoplayback/test_audio_stream",
            "uploader": "Arabic Media Channel",
            "thumbnail": "https://example.com/thumb.jpg",
            "http_headers": {
                "User-Agent": "Custom-Test-Agent"
            }
        }
        # Fake audio bytes of sufficient size (>= 1000 bytes)
        self.fake_audio_bytes = b"ID3" + b"\x00" * 2048

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_direct_pipe_success(self, mock_ydl_cls, mock_subproc_run):
        """Test successful in-memory audio extraction via direct stream pipe."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = self.sample_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        mock_proc = MagicMock()
        mock_proc.stdout = self.fake_audio_bytes
        mock_subproc_run.return_value = mock_proc

        audio_bytes, meta = self.detector.extract_audio_from_youtube(
            url=self.sample_url,
            start_seconds=60,
            duration_seconds=90
        )

        # Assertions on returned audio
        self.assertEqual(audio_bytes, self.fake_audio_bytes)

        # Assertions on returned metadata
        self.assertEqual(meta["title"], "Arabic Podcast Episode 1")
        self.assertEqual(meta["duration"], 600)
        self.assertEqual(meta["channel"], "Arabic Media Channel")
        self.assertEqual(meta["start_time"], 60)
        self.assertEqual(meta["slice_duration"], 90)
        self.assertEqual(meta["audio_size_bytes"], len(self.fake_audio_bytes))
        self.assertEqual(meta["thumbnail"], "https://example.com/thumb.jpg")

        # Verify subprocess.run was called with FFmpeg and correct streaming flags
        mock_subproc_run.assert_called_once()
        cmd = mock_subproc_run.call_args[0][0]
        self.assertIn("-ss", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "60")
        self.assertIn("-t", cmd)
        self.assertEqual(cmd[cmd.index("-t") + 1], "90")
        self.assertIn("-i", cmd)
        self.assertEqual(cmd[cmd.index("-i") + 1], self.sample_info["url"])
        self.assertIn("pipe:1", cmd)

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_auto_detect_url_timestamp(self, mock_ydl_cls, mock_subproc_run):
        """Test auto-detecting timestamp from YouTube URL parameters (e.g. ?t=45m)."""
        long_video_info = dict(self.sample_info)
        long_video_info["duration"] = 3600  # 1 hour video, so 45m (2700s) is valid

        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = long_video_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        mock_proc = MagicMock()
        mock_proc.stdout = self.fake_audio_bytes
        mock_subproc_run.return_value = mock_proc

        url_with_ts = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=45m"
        audio_bytes, meta = self.detector.extract_audio_from_youtube(
            url=url_with_ts,
            start_seconds=60  # Default start_seconds allows auto-detecting URL timestamp
        )

        # 45 minutes = 2700 seconds
        self.assertEqual(meta["start_time"], 2700)
        cmd = mock_subproc_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "2700")

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_string_timestamp_parsing(self, mock_ydl_cls, mock_subproc_run):
        """Test passing human-readable timestamps like '01:30' or '35:00' to start_seconds."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = self.sample_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        mock_proc = MagicMock()
        mock_proc.stdout = self.fake_audio_bytes
        mock_subproc_run.return_value = mock_proc

        _, meta = self.detector.extract_audio_from_youtube(
            url=self.sample_url,
            start_seconds="01:30",  # 1 min 30 sec = 90 sec
            duration_seconds=45
        )

        self.assertEqual(meta["start_time"], 90)
        cmd = mock_subproc_run.call_args[0][0]
        self.assertEqual(cmd[cmd.index("-ss") + 1], "90")

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_adjust_start_time_when_exceeding_duration(self, mock_ydl_cls, mock_subproc_run):
        """Test that start time is clamped to video_duration / 4 if requested start exceeds total length."""
        short_video_info = dict(self.sample_info)
        short_video_info["duration"] = 120  # 2 minute video

        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = short_video_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        mock_proc = MagicMock()
        mock_proc.stdout = self.fake_audio_bytes
        mock_subproc_run.return_value = mock_proc

        _, meta = self.detector.extract_audio_from_youtube(
            url=self.sample_url,
            start_seconds=300,  # 300s > 120s video duration
            duration_seconds=30
        )

        # Adjusted to duration / 4 = 120 / 4 = 30s
        self.assertEqual(meta["start_time"], 30)

    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_extract_info_failure_raises_runtime_error(self, mock_ydl_cls):
        """Test that failure to retrieve video metadata raises RuntimeError."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.side_effect = Exception("HTTP 404: Video not found")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with self.assertRaises(RuntimeError) as ctx:
            self.detector.extract_audio_from_youtube(self.sample_url)

        self.assertIn("Could not retrieve YouTube video info", str(ctx.exception))
        self.assertIn("HTTP 404", str(ctx.exception))

    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_missing_stream_url_raises_runtime_error(self, mock_ydl_cls):
        """Test that missing direct audio stream URL raises RuntimeError."""
        no_stream_info = dict(self.sample_info)
        no_stream_info["url"] = None

        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = no_stream_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        with self.assertRaises(RuntimeError) as ctx:
            self.detector.extract_audio_from_youtube(self.sample_url)

        self.assertIn("Could not find a playable direct audio stream", str(ctx.exception))

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_fallback_to_range_download_on_pipe_failure(self, mock_ydl_cls, mock_subproc_run):
        """Test that when direct pipe fails, it falls back to yt-dlp partial range download."""
        # Setup extract_info
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = self.sample_info
        
        # When fallback downloads, write fake audio to slice.mp3 in the temp directory
        def fake_download(urls):
            # Find the output template directory created in fallback
            # The second YoutubeDL instance has outtmpl in ydl_opts
            call_opts = mock_ydl_cls.call_args[0][0] if mock_ydl_cls.call_args else {}
            outtmpl = call_opts.get("outtmpl", "")
            if outtmpl:
                temp_dir = os.path.dirname(outtmpl)
                target_file = os.path.join(temp_dir, "slice.mp3")
                with open(target_file, "wb") as f:
                    f.write(self.fake_audio_bytes)

        mock_ydl.download.side_effect = fake_download
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        # Make direct pipe fail
        mock_subproc_run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"Stream error")

        audio_bytes, meta = self.detector.extract_audio_from_youtube(
            url=self.sample_url,
            start_seconds=60,
            duration_seconds=90
        )

        self.assertEqual(audio_bytes, self.fake_audio_bytes)
        self.assertEqual(meta["title"], self.sample_info["title"])
        self.assertEqual(meta["start_time"], 60)

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_both_pipe_and_fallback_failure_raises_error(self, mock_ydl_cls, mock_subproc_run):
        """Test that if direct pipe and range download both fail, RuntimeError is raised."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = self.sample_info
        mock_ydl.download.side_effect = Exception("Range download blocked")
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        mock_subproc_run.side_effect = Exception("FFmpeg connection reset")

        with self.assertRaises(RuntimeError) as ctx:
            self.detector.extract_audio_from_youtube(self.sample_url)

        self.assertIn("Audio extraction failed on both stream pipe and range download", str(ctx.exception))

    @patch("dialect_engine.subprocess.run")
    @patch("dialect_engine.yt_dlp.YoutubeDL")
    def test_empty_or_too_short_audio_raises_error(self, mock_ydl_cls, mock_subproc_run):
        """Test that audio output shorter than 1000 bytes raises RuntimeError."""
        mock_ydl = MagicMock()
        mock_ydl.extract_info.return_value = self.sample_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        # Return only 50 bytes
        mock_proc = MagicMock()
        mock_proc.stdout = b"short" * 10
        mock_subproc_run.return_value = mock_proc

        with self.assertRaises(RuntimeError) as ctx:
            self.detector.extract_audio_from_youtube(self.sample_url)

        self.assertIn("Extracted audio stream is empty or too short", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
