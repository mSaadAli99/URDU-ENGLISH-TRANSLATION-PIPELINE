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
#   6. Two-pass ASR               — doubtful segments are re-transcribed
#      with language="ur" or language="en" for code-switched interviews.
# ============================================================

import os
import sys
import math
from typing import Optional

from pipeline.utils import (
    save_json, get_interview_id, format_timestamp,
    print_banner, now_str,
    collapse_repetitions, merge_short_segments, is_urdu_text,
    is_clearly_english, is_possible_roman_urdu,
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
        return round(0.90 * word_conf + 0.10 * logprob_conf, 4)

    return logprob_conf


def _build_transcribe_kwargs(language=None, initial_prompt=None, vad_filter=True):
    """Shared faster-whisper kwargs for pass 1 and pass 2."""
    kwargs = dict(
        temperature=config.WHISPER_TEMPERATURE,
        no_speech_threshold=config.WHISPER_NO_SPEECH_THRESHOLD,
        compression_ratio_threshold=config.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=config.WHISPER_LOG_PROB_THRESHOLD,
        word_timestamps=True,
        beam_size=config.WHISPER_BEAM_SIZE,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or config.WHISPER_INITIAL_PROMPT,
        vad_filter=vad_filter,
    )
    if vad_filter:
        kwargs["vad_parameters"] = dict(
            min_silence_duration_ms=700,
            speech_pad_ms=200,
        )
    if language is not None:
        kwargs["language"] = language
    elif config.WHISPER_LANGUAGE is not None:
        kwargs["language"] = config.WHISPER_LANGUAGE
    return kwargs


def _whisper_seg_to_dict(seg, default_language="unknown"):
    """Convert a faster-whisper segment object to our pipeline dict."""
    conf = _compute_confidence(seg)
    seg_lang = getattr(seg, "language", None) or default_language
    text = seg.text.strip()

    return {
        "segment_id"    : seg.id,
        "start"         : round(seg.start, 3),
        "end"           : round(seg.end, 3),
        "start_fmt"     : format_timestamp(seg.start),
        "end_fmt"       : format_timestamp(seg.end),
        "text"          : text,
        "raw_confidence": conf,
        "confidence"    : calibrate_confidence(conf),
        "text_quality"  : score_text_quality(text),
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
    }


def _classify_pass2_action(seg: dict, global_language: Optional[str]) -> str:
    """
    Decide whether to keep pass-1 text or re-transcribe as Urdu or English.

    Returns 'keep', 'ur', or 'en'.
    """
    text = seg.get("text", "")
    lang = (seg.get("language") or "unknown").lower()
    conf = seg.get("confidence", 0.0)

    if is_urdu_text(text):
        return "keep"

    if is_clearly_english(text) and conf >= config.CONFIDENCE_THRESHOLD:
        if lang in ("en", "english"):
            return "keep"
        if lang == "unknown" and not is_possible_roman_urdu(text):
            return "keep"

    if lang == "ur" or is_possible_roman_urdu(text):
        return "ur"

    if conf < config.CONFIDENCE_THRESHOLD:
        if lang in ("unknown", "ur"):
            return "ur"
        if lang == "en":
            return "en"
        if global_language == "ur":
            return "ur"
        return "ur"

    if lang == "en" and not is_clearly_english(text):
        return "en"

    if lang == "unknown" and global_language == "en" and is_clearly_english(text):
        return "keep"

    return "keep"


def _is_pass2_better(old_seg: dict, new_text: str, new_meta: dict, target_lang: str) -> bool:
    """Return True when pass-2 output should replace pass-1 for this segment."""
    if not new_text.strip():
        return False

    margin = config.WHISPER_PASS2_CONFIDENCE_MARGIN
    old_conf = old_seg.get("confidence", 0.0)
    new_conf = new_meta.get("confidence", 0.0)
    old_text = old_seg.get("text", "")

    if target_lang == "ur":
        if is_urdu_text(new_text) and not is_urdu_text(old_text):
            return True
        if is_urdu_text(new_text) and is_urdu_text(old_text):
            return new_conf >= old_conf - margin
        if is_possible_roman_urdu(new_text) and not is_possible_roman_urdu(old_text):
            return new_conf >= old_conf - margin
        return new_conf > old_conf + margin

    if is_urdu_text(old_text) and is_clearly_english(new_text):
        return True
    if is_clearly_english(new_text):
        return new_conf >= old_conf - margin
    return new_conf > old_conf + margin


def _retranscribe_clip(model, audio_path: str, start: float, end: float, language: str):
    """Re-transcribe a single time range with a forced language."""
    prompt = (
        config.WHISPER_PASS2_PROMPT_UR
        if language == "ur"
        else config.WHISPER_PASS2_PROMPT_EN
    )
    kwargs = _build_transcribe_kwargs(
        language=language,
        initial_prompt=prompt,
        vad_filter=False,
    )
    kwargs["clip_timestamps"] = [start, end]

    segments_iter, _ = model.transcribe(audio_path, **kwargs)

    texts = []
    last_seg = None
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            texts.append(text)
            last_seg = seg

    if not texts:
        return "", {}

    combined = " ".join(texts).strip()
    if last_seg is None:
        return combined, {}

    conf = _compute_confidence(last_seg)
    return combined, {
        "confidence"    : calibrate_confidence(conf),
        "raw_confidence": conf,
        "text_quality"  : score_text_quality(combined),
        "avg_logprob"   : round(getattr(last_seg, "avg_logprob", -1.0), 4),
        "language"      : language,
        "is_urdu"       : is_urdu_text(combined),
    }


def _apply_two_pass(model, audio_path: str, segments: list, info) -> tuple[list, dict]:
    """Pass 2: re-transcribe doubtful segments with forced Urdu or English."""
    global_lang = getattr(info, "language", None)
    stats = {
        "enabled"           : True,
        "retranscribed_ur"  : 0,
        "retranscribed_en"  : 0,
        "kept_pass1"        : 0,
        "improved_ur"       : 0,
        "improved_en"       : 0,
    }

    print("\n  Pass 2: re-transcribing doubtful segments ...")
    for seg in segments:
        action = _classify_pass2_action(seg, global_lang)
        if action == "keep":
            seg["pass2_action"] = "keep"
            stats["kept_pass1"] += 1
            continue

        stats[f"retranscribed_{action}"] += 1
        new_text, new_meta = _retranscribe_clip(
            model, audio_path, seg["start"], seg["end"], action,
        )

        if _is_pass2_better(seg, new_text, new_meta, action):
            seg["text"] = new_text
            seg["language"] = new_meta.get("language", action)
            seg["is_urdu"] = new_meta.get("is_urdu", is_urdu_text(new_text))
            seg["confidence"] = new_meta.get("confidence", seg["confidence"])
            seg["raw_confidence"] = new_meta.get("raw_confidence", seg["raw_confidence"])
            seg["text_quality"] = new_meta.get("text_quality", seg["text_quality"])
            seg["avg_logprob"] = new_meta.get("avg_logprob", seg["avg_logprob"])
            seg["pass2_action"] = f"improved_{action}"
            stats[f"improved_{action}"] += 1
        else:
            seg["pass2_action"] = f"kept_pass1_over_{action}"

    print(
        f"  Pass 2 summary: "
        f"{stats['retranscribed_ur']} Urdu retries ({stats['improved_ur']} improved), "
        f"{stats['retranscribed_en']} English retries ({stats['improved_en']} improved), "
        f"{stats['kept_pass1']} kept from pass 1"
    )
    return segments, stats


# ── Main entry point ──────────────────────────────────────────

def transcribe(audio_path: str) -> dict:
    """
    Stage 1: Transcribe audio using faster-whisper.

    When WHISPER_TWO_PASS is enabled and language is auto-detect:
      Pass 1 — full-audio auto-detect transcription
      Pass 2 — doubtful segments re-transcribed with language="ur" or "en"
    """
    print_banner(1, "TRANSCRIPTION (ASR)")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    lang_display = config.WHISPER_LANGUAGE or "auto-detect"
    two_pass_enabled = (
        config.WHISPER_TWO_PASS
        and config.WHISPER_LANGUAGE is None
    )
    temp_display = (
        config.WHISPER_TEMPERATURE
        if isinstance(config.WHISPER_TEMPERATURE, (int, float))
        else f"{config.WHISPER_TEMPERATURE[0]} (fallback: {config.WHISPER_TEMPERATURE[1:]})"
    )
    print(f"  Audio file   : {audio_path}")
    print(f"  Model        : whisper-{config.WHISPER_MODEL}")
    print(f"  Language     : {lang_display}")
    print(f"  Two-pass ASR : {'on' if two_pass_enabled else 'off'}")
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

    transcribe_kwargs = _build_transcribe_kwargs()

    print(f"\n  Pass 1: transcribing audio (language: {lang_display}) ...")
    segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

    raw_segments = []
    for seg in segments_iter:
        seg_lang = getattr(seg, "language", None) or config.WHISPER_LANGUAGE or "unknown"
        raw_segments.append(_whisper_seg_to_dict(seg, default_language=seg_lang))

    raw_count = len(raw_segments)
    print(f"  Raw segments from Whisper: {raw_count}")

    merged_segments = merge_short_segments(
        raw_segments,
        min_duration=config.WHISPER_MERGE_MIN_DURATION,
        min_words=config.WHISPER_MERGE_MIN_WORDS,
    )
    merged_count = raw_count - len(merged_segments)
    if merged_count:
        print(f"  Merged {merged_count} micro-segments into neighbours.")

    before_filter = len(merged_segments)
    segments = collapse_repetitions(merged_segments, max_consecutive=config.REPETITION_MAX_CONSECUTIVE)
    removed_loops = before_filter - len(segments)
    if removed_loops:
        print(f"  Collapsed {removed_loops} repeated hallucination segments.")

    pass2_stats = {"enabled": False}
    if two_pass_enabled and segments:
        segments, pass2_stats = _apply_two_pass(model, audio_path, segments, info)

    for i, seg in enumerate(segments, start=1):
        seg["segment_id"] = i
        if "pass2_action" not in seg:
            seg["pass2_action"] = "n/a"

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
        p2   = seg.get("pass2_action", "")
        print(
            f"    [{flag}][{lang}]{mrg} {seg['start_fmt']} -> {seg['end_fmt']} "
            f"| conf={seg['confidence']:.3f} | {p2:18s} | {seg['text'][:45]}"
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
        "two_pass_asr"       : pass2_stats,
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
