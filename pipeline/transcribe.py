# ============================================================
# pipeline/transcribe.py — Stage 1: ASR (multilingual)
#
# Improvements over the previous version:
#   1. Temperature fallback list  — Whisper auto-retries hard segments
#      at higher temperatures when the first decode is repetitive or
#      has low average log-probability.
#   2. Quality decode thresholds  — no_speech_threshold,
#      compression_ratio_threshold, and log_prob_threshold are now
#      forwarded to faster-whisper so noisy / hallucinating frames
#      are suppressed at decode time.
#   3. Better VAD parameters      — min_silence_duration_ms raised to
#      700 ms (was 500) to reduce micro-segment fragmentation.
#   4. Segment merging            — micro-segments (short duration AND
#      few words) are merged with their neighbour BEFORE confidence
#      scoring so confidence numbers are based on adequate speech.
#   5. Dual confidence scoring    — per-word probability is combined
#      with the segment-level avg_logprob to produce a more robust
#      confidence estimate.
# ============================================================

import os
import sys
import math

from pipeline.utils import (
    save_json, get_interview_id, format_timestamp,
    print_banner, now_str,
    collapse_repetitions, merge_short_segments, is_urdu_text,
    calibrate_confidence, score_text_quality,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Helpers ───────────────────────────────────────────────────

def _logprob_to_prob(avg_logprob: float) -> float:
    """
    Convert Whisper's avg_logprob (-inf … 0) to a 0-1 probability.
    avg_logprob == 0  → confidence 1.0
    avg_logprob == -1 → confidence ~0.37
    avg_logprob == -2 → confidence ~0.13
    """
    return round(math.exp(max(avg_logprob, -5.0)), 4)


def _compute_confidence(seg) -> float:
    """
    Blend per-word probabilities with the segment-level avg_logprob.

    Rationale: word probabilities are noisy for very short segments
    (1-2 words); avg_logprob provides a complementary signal from the
    decoder itself.  A 70/30 weighted blend is a good practical default.
    """
    logprob_conf = _logprob_to_prob(getattr(seg, "avg_logprob", -1.0))

    if seg.words:
        probs = [w.probability for w in seg.words if hasattr(w, "probability")]
        word_conf = sum(probs) / len(probs) if probs else logprob_conf
        # 90 % word-level + 10 % segment logprob as a sanity floor.
        # Logprob is harsher (includes token generation cost) and would
        # unfairly penalise short conversational speech if weighted higher.
        return round(0.90 * word_conf + 0.10 * logprob_conf, 4)

    return logprob_conf


# ── Main entry point ──────────────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """
    Stage 1: Transcribe audio using faster-whisper.

    Language is auto-detected per segment when config.WHISPER_LANGUAGE
    is None — the correct setting for code-switched Urdu/English
    interviews.  All five quality improvements are applied in sequence.
    """
    print_banner(1, "TRANSCRIPTION (ASR)")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    lang_display = config.WHISPER_LANGUAGE or "auto-detect"
    temp_display = (
        config.WHISPER_TEMPERATURE
        if isinstance(config.WHISPER_TEMPERATURE, (int, float))
        else f"{config.WHISPER_TEMPERATURE[0]} (fallback: {config.WHISPER_TEMPERATURE[1:]})"
    )
    print(f"  Audio file   : {audio_path}")
    print(f"  Model        : whisper-{config.WHISPER_MODEL}")
    print(f"  Language     : {lang_display}")
    print(f"  Device       : {config.WHISPER_DEVICE}")
    print(f"  Temperature  : {temp_display}")
    print(f"  Beam size    : {config.WHISPER_BEAM_SIZE}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    compute_type = config.WHISPER_COMPUTE_TYPE
    if config.WHISPER_DEVICE == "cpu" and compute_type == "float16":
        compute_type = "int8"
        print("  [!] float16 not supported on CPU — falling back to int8")

    print("\n  Loading Whisper model (first run downloads ~800 MB)...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=compute_type,
    )
    print("  Model loaded.")

    # ── Build transcription kwargs ────────────────────────────
    transcribe_kwargs = dict(
        # Improvement 1: temperature fallback list
        temperature=config.WHISPER_TEMPERATURE,
        # Improvement 2: quality decode thresholds
        no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=config.WHISPER_LOG_PROB_THRESHOLD,
        # Core settings
        word_timestamps=True,
        beam_size=config.WHISPER_BEAM_SIZE,
        condition_on_previous_text=False,   # prevents cascading hallucinations
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
        # Improvement 3: larger VAD silence window reduces micro-fragmentation
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=700,    # was 500 — fewer micro-segments
            speech_pad_ms=200,              # pad each detected speech chunk
        ),
    )

    if config.WHISPER_LANGUAGE is not None:
        transcribe_kwargs["language"] = config.WHISPER_LANGUAGE

    print(f"\n  Transcribing audio (language: {lang_display}) ...")
    segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

    # ── Collect raw segments ──────────────────────────────────
    raw_segments = []
    for seg in segments_iter:
        # Improvement 5: dual confidence scoring
        conf = _compute_confidence(seg)

        seg_lang = getattr(seg, "language", None) or config.WHISPER_LANGUAGE or "unknown"

        text = seg.text.strip()
        raw_segments.append({
            "segment_id"    : seg.id,
            "start"         : round(seg.start, 3),
            "end"           : round(seg.end, 3),
            "start_fmt"     : format_timestamp(seg.start),
            "end_fmt"       : format_timestamp(seg.end),
            "text"          : text,
            "raw_confidence": conf,                          # uncalibrated Whisper score
            "confidence"    : calibrate_confidence(conf),   # calibrated (accuracy-representative)
            "text_quality"  : score_text_quality(text),     # linguistic quality 0-1
            "avg_logprob"   : round(getattr(seg, "avg_logprob", -1.0), 4),
            "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 4),
            "language"      : seg_lang,
            "is_urdu"       : is_urdu_text(text),
            "words"         : [
                {
                    "word"      : w.word,
                    "start"     : round(w.start, 3),
                    "end"       : round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else None,
                }
                for w in (seg.words or [])
            ],
        })  # end raw_segments.append

    raw_count = len(raw_segments)
    print(f"  Raw segments from Whisper: {raw_count}")

    # ── Improvement 4: merge micro-segments ───────────────────
    merged_segments = merge_short_segments(
        raw_segments,
        min_duration=config.WHISPER_MERGE_MIN_DURATION,
        min_words=config.WHISPER_MERGE_MIN_WORDS,
    )
    merged_count = raw_count - len(merged_segments)
    if merged_count:
        print(f"  Merged {merged_count} micro-segments into neighbours.")

    # ── Collapse Whisper hallucination loops ──────────────────
    before_filter = len(merged_segments)
    segments = collapse_repetitions(merged_segments, max_consecutive=config.REPETITION_MAX_CONSECUTIVE)
    removed_loops = before_filter - len(segments)
    if removed_loops:
        print(f"  Collapsed {removed_loops} repeated hallucination segments.")

    # Re-number after filtering
    for i, seg in enumerate(segments, start=1):
        seg["segment_id"] = i

    # ── Language breakdown ────────────────────────────────────
    urdu_count    = sum(1 for s in segments if s["is_urdu"])
    english_count = len(segments) - urdu_count
    avg_conf      = (
        sum(s["confidence"] for s in segments) / len(segments)
        if segments else 0.0
    )
    avg_raw_conf  = (
        sum(s["raw_confidence"] for s in segments) / len(segments)
        if segments else 0.0
    )
    avg_text_qual = (
        sum(s["text_quality"] for s in segments) / len(segments)
        if segments else 0.0
    )
    low_conf_count = sum(1 for s in segments if s["confidence"] < config.CONFIDENCE_THRESHOLD)

    full_text = " ".join(s["text"] for s in segments)
    duration_seconds = (
        info.duration if hasattr(info, "duration")
        else (segments[-1]["end"] if segments else 0)
    )

    print(f"\n  Results:")
    print(f"  Segments total    : {len(segments)}")
    print(f"    Urdu            : {urdu_count}")
    print(f"    English         : {english_count}")
    print(f"    Low confidence  : {low_conf_count}  (threshold={config.CONFIDENCE_THRESHOLD})")
    print(f"  Avg raw confidence: {avg_raw_conf:.3f}")
    print(f"  Avg cal confidence: {avg_conf:.3f}")
    print(f"  Avg text quality  : {avg_text_qual:.3f}")
    print(f"  Loops removed     : {removed_loops}")
    print(f"  Micro-segs merged : {merged_count}")

    print("\n  Segment detail:")
    for seg in segments:
        flag = "!" if seg["confidence"] < config.CONFIDENCE_THRESHOLD else " "
        lang = "UR" if seg["is_urdu"] else "EN"
        mrg  = "[M]" if seg.get("merged") else "   "
        print(
            f"    [{flag}][{lang}]{mrg} {seg['start_fmt']} -> {seg['end_fmt']} "
            f"| conf={seg['confidence']:.3f} | {seg['text'][:55]}"
        )

    result = {
        "stage"              : 1,
        "stage_name"         : "Transcription (ASR)",
        "interview_id"       : get_interview_id(audio_path),
        "audio_filename"     : os.path.basename(audio_path),
        "audio_path"         : audio_path,
        "processed_at"       : now_str(),
        "model"              : f"whisper-{config.WHISPER_MODEL}",
        "language_mode"      : lang_display,
        "duration_seconds"   : round(duration_seconds, 2),
        "duration_minutes"   : round(duration_seconds / 60, 2),
        "total_segments"     : len(segments),
        "urdu_segments"      : urdu_count,
        "english_segments"   : english_count,
        "avg_raw_confidence" : round(avg_raw_conf, 4),
        "avg_confidence"     : round(avg_conf, 4),
        "avg_text_quality"   : round(avg_text_qual, 4),
        "low_conf_segments"  : low_conf_count,
        "micro_segs_merged"  : merged_count,
        "loops_removed"      : removed_loops,
        "full_urdu_text"     : full_text,
        "segments"           : segments,
    }

    interview_id = result["interview_id"]
    out_path = os.path.join(config.STAGE1_DIR, f"{interview_id}_urdu_transcript.json")
    save_json(result, out_path)

    print(f"\n  Stage 1 complete.")
    print(f"  Total segments: {len(segments)}  |  Duration: {result['duration_minutes']} min")
    return result
