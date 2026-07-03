# ============================================================
# pipeline/verify_transcript.py — Stage 2: Transcript Verification
#
# Quality scoring redesigned with four independent factors:
#
#   Factor 1 — Calibrated confidence (40 %)
#       Whisper's raw word probabilities are conservative. Using
#       calibrated values (stored on each segment by Stage 1) gives a
#       fairer picture of actual word accuracy.
#
#   Factor 2 — Linguistic text quality (35 %)
#       Independently assess whether the output text is coherent English
#       (or Urdu). High-quality text should score high even when Whisper
#       was uncertain about specific tokens.
#
#   Factor 3 — Segment coverage (15 %)
#       Fraction of segments passing the confidence threshold. Rewards
#       audio where most segments are solid.
#
#   Factor 4 — Cleanliness bonus (10 %)
#       Full marks for zero hallucination loops, zero merged micro-segs
#       (i.e., Whisper produced clean, well-sized segments natively).
#       Partial marks when the pipeline had to repair the output.
# ============================================================

import os
import sys
from pipeline.utils import save_json, print_banner, now_str, score_text_quality

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def verify_transcript(stage1_result: dict) -> dict:
    """
    Stage 2: Multi-factor quality assessment of the ASR output.
    """
    print_banner(2, "VERIFICATION OF TRANSCRIPT")

    segments     = stage1_result["segments"]
    interview_id = stage1_result["interview_id"]
    threshold    = config.CONFIDENCE_THRESHOLD

    loops_removed    = stage1_result.get("loops_removed", 0)
    micro_merged     = stage1_result.get("micro_segs_merged", 0)

    print(f"  Interview ID        : {interview_id}")
    print(f"  Total segments      : {len(segments)}")
    print(f"  Urdu segments       : {stage1_result.get('urdu_segments', '?')}")
    print(f"  English segments    : {stage1_result.get('english_segments', '?')}")
    print(f"  Micro-segs merged   : {micro_merged}")
    print(f"  Loops removed       : {loops_removed}")
    print(f"  Avg raw confidence  : {stage1_result.get('avg_raw_confidence', '?')}")
    print(f"  Avg cal confidence  : {stage1_result.get('avg_confidence', '?')}")
    print(f"  Avg text quality    : {stage1_result.get('avg_text_quality', '?')}")
    print(f"  Confidence threshold: {threshold}")

    verified_segments = []
    flagged_segments  = []
    cal_confidences   = []
    text_qualities    = []

    for seg in segments:
        # Use calibrated confidence stored by Stage 1; fall back to raw if missing
        cal_conf  = seg.get("confidence", seg.get("raw_confidence", 0.0))
        raw_conf  = seg.get("raw_confidence", cal_conf)
        is_low    = bool(cal_conf < threshold)

        # Text quality: use stored value if present, otherwise compute now
        tq = seg.get("text_quality")
        if tq is None:
            tq = score_text_quality(seg.get("text", ""))

        cal_confidences.append(cal_conf)
        text_qualities.append(tq)

        verified_seg = {
            **seg,
            "text_quality"  : tq,
            "verified_text" : (
                f"[LOW CONFIDENCE] {seg['text']}".strip() if is_low else seg["text"]
            ),
            "flagged"       : bool(is_low),
            "flag_reason"   : (
                f"Calibrated confidence {cal_conf:.2f} < threshold {threshold}"
                if is_low else None
            ),
        }

        verified_segments.append(verified_seg)

        if is_low:
            flagged_segments.append({
                "segment_id"    : seg["segment_id"],
                "start_fmt"     : seg["start_fmt"],
                "end_fmt"       : seg["end_fmt"],
                "text"          : seg["text"],
                "raw_confidence": raw_conf,
                "confidence"    : cal_conf,
                "text_quality"  : tq,
                "language"      : seg.get("language", "unknown"),
                "reason"        : verified_seg["flag_reason"],
            })

        icon = "! FLAGGED" if is_low else "ok"
        lang = "UR" if seg.get("is_urdu") else "EN"
        mrg  = "[M]" if seg.get("merged") else "   "
        print(
            f"    [{icon:8}][{lang}]{mrg} seg {seg['segment_id']:03d} "
            f"| conf={cal_conf:.3f} | tq={tq:.2f} | {seg['text'][:45]}"
        )

    # ── Four-factor quality score ─────────────────────────────
    n = len(segments)

    # Factor 1: average calibrated confidence (0-1)
    f1_conf = sum(cal_confidences) / n if n else 0.0

    # Factor 2: average linguistic text quality (0-1)
    f2_text = sum(text_qualities) / n if n else 0.0

    # Factor 3: fraction of segments above threshold (0-1)
    f3_cov  = sum(1 for c in cal_confidences if c >= threshold) / n if n else 0.0

    # Factor 4: cleanliness bonus (0-1)
    #   - 0 loops AND 0 micro-merges → 1.0 (Whisper was perfectly clean)
    #   - loops/merges reduce the bonus proportionally
    loop_hit  = min(loops_removed * 0.10, 0.50)
    merge_hit = min(micro_merged  * 0.03, 0.30)
    f4_clean  = max(0.0, 1.0 - loop_hit - merge_hit)

    # Weighted combination
    quality_score = round(
        (f1_conf * 0.40 + f2_text * 0.35 + f3_cov * 0.15 + f4_clean * 0.10) * 100,
        1,
    )

    verified_full_text = " ".join(s["verified_text"] for s in verified_segments)

    verification_report = {
        "total_segments"      : n,
        "flagged_segments"    : len(flagged_segments),
        "good_segments"       : n - len(flagged_segments),
        "avg_calibrated_conf" : round(f1_conf, 4),
        "avg_text_quality"    : round(f2_text, 4),
        "pct_good_segments"   : round(f3_cov, 4),
        "cleanliness_score"   : round(f4_clean, 4),
        "quality_score"       : quality_score,
        "quality_label"       : _quality_label(quality_score),
        "duration_covered_min": stage1_result["duration_minutes"],
        "confidence_threshold": threshold,
        "urdu_segments"       : stage1_result.get("urdu_segments", 0),
        "english_segments"    : stage1_result.get("english_segments", 0),
        "micro_segs_merged"   : micro_merged,
        "loops_removed"       : loops_removed,
        "flagged_details"     : flagged_segments,
    }

    print(f"\n  ── Verification Report ──────────────────────────")
    print(f"  Factor 1 — Calibrated confidence : {f1_conf:.4f}  (x0.40 = {f1_conf*0.40:.4f})")
    print(f"  Factor 2 — Text linguistic quality: {f2_text:.4f}  (x0.35 = {f2_text*0.35:.4f})")
    print(f"  Factor 3 — Coverage (pct good)    : {f3_cov:.4f}  (x0.15 = {f3_cov*0.15:.4f})")
    print(f"  Factor 4 — Cleanliness bonus      : {f4_clean:.4f}  (x0.10 = {f4_clean*0.10:.4f})")
    print(f"  -----------------------------------------------")
    print(f"  Quality score  : {quality_score}/100 ({_quality_label(quality_score)})")
    print(f"  Flagged segs   : {len(flagged_segments)} / {n}")

    if quality_score < config.MIN_QUALITY_SCORE:
        print(f"\n  [!] Quality {quality_score} is below minimum {config.MIN_QUALITY_SCORE}.")

    result = {
        "stage"                : 2,
        "stage_name"           : "Verification of Transcript",
        "interview_id"         : interview_id,
        "audio_filename"       : stage1_result["audio_filename"],
        "processed_at"         : now_str(),
        "original_full_text"   : stage1_result["full_urdu_text"],
        "verified_full_text"   : verified_full_text,
        "quality_score"        : quality_score,
        "quality_label"        : _quality_label(quality_score),
        "verification_report"  : verification_report,
        "verified_segments"    : verified_segments,
        "duration_minutes"     : stage1_result["duration_minutes"],
        "audio_path"           : stage1_result["audio_path"],
    }

    out_path = os.path.join(config.STAGE2_DIR, f"{interview_id}_verified_transcript.json")
    save_json(result, out_path)

    print(f"\n  Stage 2 complete.")
    return result


def _quality_label(score: float) -> str:
    if score >= 88: return "Excellent"
    if score >= 75: return "Good"
    if score >= 60: return "Fair"
    if score >= 45: return "Poor"
    return "Very Poor"
