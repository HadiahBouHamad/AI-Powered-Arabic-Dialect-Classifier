"""
CLI Utility for Arabic Dialect Detection on YouTube Videos
Usage:
    python cli_dialect.py --url "https://www.youtube.com/watch?v=..." --start 60 --duration 90
"""

import argparse
import sys
from dialect_engine import DialectDetector

def main():
    parser = argparse.ArgumentParser(description="Identify Arabic Dialects and Per-Speaker Percentages from a YouTube link.")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--start", type=str, default="01:00", help="Start offset (e.g. '35:00', '1:15:00', '45m', or seconds)")
    parser.add_argument("--duration", type=int, default=90, help="Duration of audio slice in seconds (default: 90)")
    parser.add_argument("--model", type=str, default="auto", help="Gemini model (default: auto - uses latest working Gemini model)")

    args = parser.parse_args()

    detector = DialectDetector(model_name=args.model)
    start_sec = DialectDetector.parse_time_offset(args.start)
    start_fmt = DialectDetector.format_timestamp(start_sec)
    end_fmt = DialectDetector.format_timestamp(start_sec + args.duration)

    print("\n" + "="*60)
    print("  مُشخِّص اللهجات والمتحدثين | Arabic Dialect & Speaker Identifier")
    print("="*60)
    print(f"Target URL: {args.url}")
    print(f"Sampling: {start_fmt} to {end_fmt} ({start_sec}s to {start_sec + args.duration}s • {args.duration}s slice)")
    print(f"Model: {detector.model_name}")
    print("="*60 + "\n")

    print("[1/2] Slicing YouTube audio directly in-memory...")
    try:
        audio_bytes, meta = detector.extract_audio_from_youtube(
            url=args.url,
            start_seconds=start_sec,
            duration_seconds=args.duration
        )
        print(f"  Title: {meta['title']}")
        print(f"  Channel: {meta['channel']}")
        print(f"  Audio Slice Size: {meta['audio_size_bytes'] / 1024:.1f} KB\n")
    except Exception as e:
        print(f"Error extracting audio: {e}", file=sys.stderr)
        sys.exit(1)

    print("[2/2] Streaming dialect identification from Gemini Flash...")
    print("-" * 60)
    full_output = []
    try:
        for chunk in detector.stream_dialect_identification(audio_bytes):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_output.append(chunk)
        print("\n" + "-" * 60)
    except Exception as e:
        print(f"\nError analyzing dialect: {e}", file=sys.stderr)
        sys.exit(1)

    parsed = detector.parse_analysis_result("".join(full_output))
    print("\n" + "="*60)
    print("  Analysis Summary:")
    print(f"  -> Dominant Dialect : {parsed['dominant_dialect']}")
    print(f"  -> Confidence       : {parsed['confidence']}")
    if parsed.get("speakers"):
        print("  -> Speakers Breakdown:")
        for spk in parsed["speakers"]:
            note_str = f" - {spk['note']}" if spk.get('note') else ""
            print(f"       • {spk['speaker']} ({spk['dialect']}): {spk['percentage']:.0f}%{note_str}")
    if parsed.get("dialect_distribution"):
        print("  -> Dialect Distribution:")
        for dialect, pct in parsed['dialect_distribution'].items():
            print(f"       • {dialect}: {pct:.0f}%")
    print("="*60)

if __name__ == "__main__":
    main()
