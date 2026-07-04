#!/usr/bin/env python3
"""
scripts/download_audio.py — Download a sample Urdu audio clip from YouTube.

Usage:
    python scripts/download_audio.py                        # download default sample
    python scripts/download_audio.py --url URL              # custom YouTube URL
    python scripts/download_audio.py --url URL --start 1275 --duration 300

Requirements:
    pip install yt-dlp
    (ffmpeg must be on PATH — already available in Colab and most Linux installs)
"""

import argparse
import os
import subprocess
import sys


DEFAULT_URL      = "https://youtu.be/pHZHYWe8Mkc"
DEFAULT_START    = 1275     # seconds into the video (21:15)
DEFAULT_DURATION = 300      # seconds to keep (5 min → ends at 1575s / 26:15)
OUTPUT_FILE      = "audio/test_audio.mp3"
FULL_FILE        = "audio/full.mp3"


def check_dependency(cmd: str, install_hint: str):
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"  ✗ '{cmd}' not found. {install_hint}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download sample Urdu audio from YouTube")
    parser.add_argument("--url",      default=DEFAULT_URL,      help="YouTube URL")
    parser.add_argument("--start",    default=DEFAULT_START,    type=int,
                        help=f"Start offset in seconds (default {DEFAULT_START})")
    parser.add_argument("--duration", default=DEFAULT_DURATION, type=int,
                        help=f"Duration in seconds (default {DEFAULT_DURATION})")
    parser.add_argument("--output",   default=OUTPUT_FILE,      help="Output file path")
    args = parser.parse_args()

    print("=" * 56)
    print("  Urdu Pipeline — Audio Downloader")
    print("=" * 56)

    check_dependency("yt-dlp",  "Run: pip install yt-dlp")
    check_dependency("ffmpeg",  "Install ffmpeg: https://ffmpeg.org/download.html")

    os.makedirs("audio", exist_ok=True)

    # Step 1: Download full audio
    print(f"\n  Source   : {args.url}")
    print(f"  Downloading full audio...")
    result = subprocess.run(
        ["yt-dlp", "-x", "--audio-format", "mp3",
         "-o", FULL_FILE, args.url],
        check=False,
    )
    if result.returncode != 0:
        print("  ✗ yt-dlp download failed. Check the URL and your network.")
        sys.exit(1)

    # Step 2: Trim clip
    print(f"\n  Trimming: start={args.start}s  duration={args.duration}s")
    trim_result = subprocess.run(
        ["ffmpeg", "-y", "-i", FULL_FILE,
         "-ss", str(args.start),
         "-t",  str(args.duration),
         "-c",  "copy", args.output],
        check=False,
    )
    if trim_result.returncode != 0:
        print("  ✗ ffmpeg trim failed. Keeping full audio instead.")
        os.rename(FULL_FILE, args.output)
    else:
        os.remove(FULL_FILE)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\n  ✔ Audio ready: {args.output}  ({size_mb:.1f} MB)")
    print(f"\n  Run the pipeline with:")
    print(f"    python main.py {args.output}")


if __name__ == "__main__":
    main()
