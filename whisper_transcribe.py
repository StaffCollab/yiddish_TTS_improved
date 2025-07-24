#!/usr/bin/env python
"""
transcribe_yiddish_whisperx.py

Transcribe Yiddish (or other minority‑language) audio and produce word‑level timestamps
using OpenAI Whisper (via faster‑whisper) and optional forced alignment with WhisperX.

Usage
-----
    python transcribe_yiddish_whisperx.py <audio-file> \
        --model large-v3 \
        --json-out yid_transcript.json \
        --srt-out yid_words.srt

Options
-------
  --model      Whisper checkpoint to use (default: large-v3). Any size/path accepted.
  --device     cuda | cpu | auto‑detected (default tries CUDA, falls back to CPU).
  --json-out   Path to write verbose JSON (segments + words + timings).
  --srt-out    Path to write an SRT with *per‑word* subtitles.
  --no-align   Skip WhisperX alignment step (use raw Whisper timings).

Dependencies
------------
    pip install faster-whisper  # Whisper inference (2–3× faster on GPU)
    pip install whisperx        # Forced alignment & diarization (optional)
    pip install srt             # Subtitle export helper

The input audio should be 16‑kHz mono WAV/MP3/FLAC. Convert with:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 out.wav
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import torch
import srt

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    sys.exit("Missing dependency: pip install faster-whisper ‑‑upgrade")

# -----------------------------------------------------------------------------
# Whisper transcription helpers
# -----------------------------------------------------------------------------

def load_whisper(model_name: str, device: str) -> WhisperModel:
    """Return an initialized Whisper model on the requested device."""
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(model: WhisperModel, audio_fp: str, language: str = "yi") -> Dict[str, Any]:
    """Run Whisper and return the raw transcription dict (segments + word timings)."""
    segments, _info = model.transcribe(
        audio_fp,
        language=language,                # ISO‑639‑1 code for Yiddish (FIXED: uncommented)
        beam_size=5,
        vad_filter=True,                  # split on voice activity to keep context short
        word_timestamps=True,
        condition_on_previous_text=False, # make decoding chunk‑independent
        suppress_tokens=[]                # don't hide any acoustically‑confident tokens
    )

    seg_dicts: List[Dict[str, Any]] = [seg._asdict() for seg in segments]
    return {
        "segments": seg_dicts,
        "text": " ".join(word.word for seg in segments for word in seg.words),
    }

# -----------------------------------------------------------------------------
# WhisperX alignment (optional but strongly recommended for ±10 ms accuracy)
# -----------------------------------------------------------------------------


def align_with_whisperx(whisper_result: Dict[str, Any], audio_fp: str, *,
                        language_code: str = "yi", device: str = "cuda") -> Dict[str, Any]:
    """Force‑align Whisper words to the waveform using WhisperX (Wav2Vec 2.0)."""
    try:
        import whisperx  # type: ignore
    except ImportError:
        raise RuntimeError("whisperx not installed – run `pip install whisperx` or use --no-align")

    # Try Yiddish first, fall back to German if no Yiddish model available
    try:
        align_model, metadata = whisperx.load_align_model(device=device, language_code=language_code)
    except Exception as e:
        print(f"[WARNING] Yiddish alignment model not available, falling back to German: {e}")
        align_model, metadata = whisperx.load_align_model(device=device, language_code="de")
    
    aligned = whisperx.align(
        whisper_result["segments"],
        align_model,
        metadata,
        audio_fp,
        device=device,
        return_char_alignments=False,
    )
    return aligned

# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------


def export_json(result: Dict[str, Any], path: pathlib.Path) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


def export_srt(result: Dict[str, Any], path: pathlib.Path) -> None:
    """Write per‑word subtitles in SRT format."""
    subs = []
    idx = 1
    for seg in result["segments"]:
        for w in seg["words"]:
            subs.append(
                srt.Subtitle(
                    index=idx,
                    start=srt.timedelta(seconds=w["start"]),
                    end=srt.timedelta(seconds=w["end"]),
                    content=w["word"],
                )
            )
            idx += 1
    path.write_text(srt.compose(subs), encoding="utf-8")

# -----------------------------------------------------------------------------
# Main CLI entry point
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe Yiddish audio and generate per‑word timestamps using Whisper + WhisperX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio", help="Path to 16‑kHz mono audio (WAV/MP3/FLAC)")
    parser.add_argument("--model", default="large-v3", help="Whisper checkpoint or local path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Computation device")
    parser.add_argument("--json-out", dest="json_out", default=None,
                        help="Write full transcript JSON to this file")
    parser.add_argument("--srt-out", dest="srt_out", default=None,
                        help="Write per‑word SRT subtitles to this file")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip WhisperX alignment (use Whisper timings only)")
    args = parser.parse_args()

    # Load Whisper
    model = load_whisper(args.model, args.device)

    # First‑pass transcription
    whisper_res = transcribe(model, args.audio)

    # Optional alignment
    final_res = whisper_res
    if not args.no_align:
        print("[+] Running WhisperX alignment …", file=sys.stderr)
        final_res = align_with_whisperx(whisper_res, args.audio, device=args.device)

    # Output
    if args.json_out:
        export_json(final_res, pathlib.Path(args.json_out))
        print(f"[✓] JSON written to {args.json_out}")

    if args.srt_out:
        export_srt(final_res, pathlib.Path(args.srt_out))
        print(f"[✓] SRT written to {args.srt_out}")

    print("[✓] Done.")


if __name__ == "__main__":
    main()
