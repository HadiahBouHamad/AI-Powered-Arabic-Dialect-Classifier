"""
CLI Utility for Arabic Dialect Detection on YouTube Videos
Usage:
    python cli_dialect.py --url "https://www.youtube.com/watch?v=..." --start 60 --duration 90
"""

import argparse
import sys
from dialect_engine import DialectDetector

def main():
    parser = argparse.ArgumentParser(description="Identify Arabic Dialects (Saudi, Bahraini, Lebanese, MSA, Other) from a YouTube link.")
    parser.add_argument("--url", type=str, required=True, help="YouTube video URL")
    parser.add_argument("--start", type=int, default=60, help="Start offset in seconds (default: 60)")
    parser.add_argument("--duration", type=int, default=90, help="Duration of audio slice in seconds (default: 90)")
    parser.add_argument("--model", type=str, default="gemini-3.1-flash-lite", help="Gemini Flash model")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("  مُشخِّص اللهجات العربية | Arabic Dialect Identifier")
    print("="*60)
    print(f"Target URL: {args.url}")
    print(f"Sampling: {args.start}s to {args.start + args.duration}s ({args.duration}s duration)")
    print(f"Model: {args.model}")
    print("="*60 + "\n")

    detector = DialectDetector(model_name=args.model)

    print("[1/2] Slicing YouTube audio directly in-memory...")
    try:
        audio_bytes, meta = detector.extract_audio_from_youtube(
            url=args.url,
            start_seconds=args.start,
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
    print("\nSummary Summary:")
    print(f"  -> Dominant Dialect: {parsed['dominant_dialect']}")
    print(f"  -> Confidence: {parsed['confidence']}")
    print("  -> Distribution:")
    for dialect, pct in parsed['distribution_raw'].items():
        print(f"       {dialect:10}: {pct}%")
    print("="*60)

if __name__ == "__main__":
    main()
