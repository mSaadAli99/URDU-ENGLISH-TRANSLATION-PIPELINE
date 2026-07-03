# ============================================================
# pipeline/translate.py — Stage 3: Translation (smart routing)
# Urdu segments  → translated via NLLB-200 (GPU) / OPUS (CPU)
# English segments → passed through unchanged (no translation needed)
# ============================================================

import os
import sys
from pipeline.utils import (
    save_json, split_sentences, chunk_sentences,
    fix_translation_errors, is_urdu_text, should_translate_segment,
    print_banner, now_str,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def translate(stage2_result: dict) -> dict:
    """
    Stage 3: Translate Urdu segments to English; pass English segments unchanged.

    This is the correct approach for code-switched interviews. Sending already-
    English text through an Urdu→English model produces nonsense.
    """
    print_banner(3, "TRANSLATION: URDU → ENGLISH (smart routing)")

    interview_id = stage2_result["interview_id"]

    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        raise ImportError("transformers / torch not installed. Run: pip install transformers torch")

    use_cuda   = config.WHISPER_DEVICE == "cuda" and torch.cuda.is_available()
    model_name = config.TRANSLATION_MODEL if use_cuda else config.CPU_TRANSLATION_MODEL
    is_nllb    = "nllb" in model_name.lower()

    size_hint = "~2.5GB" if "1.3B" in model_name else ("~1.2GB" if "600M" in model_name else "~300MB")
    print(f"  Interview ID   : {interview_id}")
    print(f"  Model          : {model_name}")
    print(f"  Routing        : Urdu→NLLB  |  English→pass-through")
    # ── Partition segments by language ───────────────────────
    verified_segments = stage2_result["verified_segments"]
    total             = len(verified_segments)

    urdu_indices  = []   # segments that need translation
    urdu_texts    = []

    for i, seg in enumerate(verified_segments):
        # Prefer the is_urdu flag from Stage 1; fall back to text analysis
        if should_translate_segment(seg["text"], seg.get("is_urdu", False)):
            urdu_indices.append(i)
            urdu_texts.append(seg["text"])

    english_count = total - len(urdu_indices)
    print(f"\n  Routing: {len(urdu_indices)} Urdu segments -> translate  |  "
          f"{english_count} English segments -> pass-through")

    # ── Early-exit optimisation: no Urdu segments → skip model load ──
    translated_urdu: dict = {}   # index → translated text

    if not urdu_indices:
        print("\n  [opt] All segments are English — skipping translation model load.")
    else:
        # Only load the model when there is actually something to translate
        print(f"\n  Loading translation model ({size_hint} first run)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device    = "cuda" if use_cuda else "cpu"
        dtype     = torch.float16 if use_cuda else torch.float32
        model_    = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
        model_.eval()
        print(f"  Model loaded on {device}.")

        tgt_lang_id = tokenizer.convert_tokens_to_ids(config.NLLB_TGT_LANG) if is_nllb else None

        def _translate_batch(texts: list) -> list:
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            gen_kwargs = dict(
                max_new_tokens=256,
                num_beams=3 if device == "cpu" else 5,
                early_stopping=True,
            )
            if tgt_lang_id is not None:
                gen_kwargs["forced_bos_token_id"] = tgt_lang_id

            with torch.no_grad():
                outputs = model_.generate(**inputs, **gen_kwargs)

            return [
                fix_translation_errors(t)
                for t in tokenizer.batch_decode(outputs, skip_special_tokens=True)
            ]

        # ── Batch translate all Urdu segments ─────────────────
        print(f"\n  Translating Urdu segments...")
        batch_size = config.BATCH_TRANSLATION_SIZE

        for start in range(0, len(urdu_indices), batch_size):
            batch_idx   = urdu_indices[start : start + batch_size]
            batch_texts = urdu_texts[start : start + batch_size]
            try:
                results = _translate_batch(batch_texts)
            except Exception as e:
                print(f"    Batch failed: {e}")
                results = ["[TRANSLATION ERROR]"] * len(batch_texts)
            for idx, eng in zip(batch_idx, results):
                translated_urdu[idx] = eng

    # ── Assemble final segment list ───────────────────────────
    print(f"\n  Assembling translated segments:")
    translated_segments = []
    for i, seg in enumerate(verified_segments):
        if i in translated_urdu:
            eng_text = translated_urdu[i]
            route    = "TRANSLATED"
        else:
            eng_text = seg["text"]   # English — use as-is
            route    = "PASS-THROUGH"

        translated_segments.append({**seg, "english_text": eng_text, "translation_route": route})

        icon = "T" if route == "TRANSLATED" else "-"
        print(f"    [{icon}] seg {seg['segment_id']:03d}/{total} | {route}"
              f" | {eng_text[:60]}...")

    full_english_text     = " ".join(s["english_text"] for s in translated_segments)
    verified_full_text    = stage2_result.get("verified_full_text", "")

    result = {
        "stage"                : 3,
        "stage_name"           : "Translation: Urdu → English",
        "interview_id"         : interview_id,
        "audio_filename"       : stage2_result["audio_filename"],
        "processed_at"         : now_str(),
        "translation_model"    : model_name,
        "urdu_full_text"       : verified_full_text,
        "english_full_text"    : full_english_text,
        "transcript_quality"   : stage2_result["quality_score"],
        "duration_minutes"     : stage2_result["duration_minutes"],
        "audio_path"           : stage2_result["audio_path"],
        "translated_segments"  : translated_segments,
        "segments_translated"  : len(urdu_indices),
        "segments_passthrough" : english_count,
        "transcript_verification_report": stage2_result["verification_report"],
    }

    out_path = os.path.join(config.STAGE3_DIR, f"{interview_id}_english_translation.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 3 complete.")
    print(f"  Translated: {len(urdu_indices)}  |  Pass-through: {english_count}")
    return result
