# ============================================================
# pipeline/deidentify.py — Stage 5: De-identification
# Uses Microsoft Presidio + spaCy to remove PII
# Replaces names, phones, emails, locations, orgs, dates, IDs
# ============================================================

import os
from pipeline.utils import save_json, print_banner, now_str

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Suppress Presidio warnings about unsupported language recognizers
import logging
logging.getLogger('presidio-analyzer').setLevel(logging.ERROR)
logging.getLogger('presidio-anonymizer').setLevel(logging.ERROR)


def deidentify(stage4_result: dict) -> dict:
    """
    Stage 5: Remove personally identifiable information (PII) from English text.

    Args:
        stage4_result: Output dict from Stage 4 (verify_translation.py)

    Returns:
        dict with de-identified text and list of removed entities
    """
    print_banner(5, "DE-IDENTIFICATION OF DATASET")

    interview_id = stage4_result["interview_id"]
    print(f"  Interview ID : {interview_id}")

    # ── Load Presidio ─────────────────────────────────────────
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        raise ImportError(
            "presidio not installed.\n"
            "Run: pip install presidio-analyzer presidio-anonymizer\n"
            "     python -m spacy download en_core_web_lg"
        )

    print("  Loading Presidio + spaCy (first run may take a moment)...")
    analyzer  = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    print("  ✔ Presidio loaded.")

    # ── Build operator config (entity → replacement tag) ──────
    operators = {
        entity_type: OperatorConfig("replace", {"new_value": tag})
        for entity_type, tag in config.ENTITY_REPLACEMENTS.items()
    }

    # ── De-identification helper ──────────────────────────────
    def _is_generic_temporal(entity_text: str) -> bool:
        """
        Return True for generic temporal words that Presidio incorrectly
        flags as DATE_TIME PII (e.g. "today", "night", "a day").
        These carry no identifying information and must not be removed.
        """
        normalised = entity_text.strip().lower()
        return normalised in config.GENERIC_TEMPORAL_WORDS

    def deidentify_text(text: str) -> tuple[str, list]:
        """
        Remove PII from a single text string.
        Returns (cleaned_text, list_of_removed_entities).

        Filters out false-positive DATE_TIME detections (generic words
        like "today", "night", "a day") before anonymizing.
        """
        if not text or not text.strip():
            return text, []

        try:
            results = analyzer.analyze(
                text=text,
                language="en",
                entities=list(config.ENTITY_REPLACEMENTS.keys()),
            )

            if not results:
                return text, []

            # Filter out generic temporal false positives
            filtered = []
            skipped = []
            for r in results:
                entity_text = text[r.start:r.end]
                if r.entity_type == "DATE_TIME" and _is_generic_temporal(entity_text):
                    skipped.append(entity_text)
                    continue
                filtered.append(r)

            if skipped:
                print(f"      [skip] Ignored generic temporal words: {skipped}")

            if not filtered:
                return text, []

            found_entities = []
            for r in filtered:
                entity_text = text[r.start:r.end]
                tag = config.ENTITY_REPLACEMENTS.get(r.entity_type, f"[{r.entity_type}]")
                found_entities.append({
                    "original"     : entity_text,
                    "type"         : r.entity_type,
                    "replaced_with": tag,
                    "score"        : round(r.score, 3),
                })

            anonymized = anonymizer.anonymize(
                text=text,
                analyzer_results=filtered,
                operators=operators,
            )

            return anonymized.text, found_entities

        except Exception as e:
            print(f"    De-identification error: {e}")
            return text, []

    # ── De-identify full English text ─────────────────────────
    print("\n  De-identifying full English text...")
    verified_english_full = stage4_result.get("verified_english_full", "")
    deidentified_full, all_entities_full = deidentify_text(verified_english_full)

    # ── De-identify segment by segment ───────────────────────
    print("\n  De-identifying segments:")
    deidentified_segments = []
    all_entities_removed  = []

    for seg in stage4_result["verified_segments"]:
        eng_text = seg.get("verified_english", seg.get("english_text", ""))
        clean_text, entities = deidentify_text(eng_text)

        deidentified_seg = {
            **seg,
            "deidentified_english" : clean_text,
            "entities_removed"     : entities,
        }
        deidentified_segments.append(deidentified_seg)
        all_entities_removed.extend(entities)

        entity_count = len(entities)
        icon = "[PII]" if entity_count > 0 else "[ok ]"
        print(f"    {icon} seg {seg['segment_id']:03d} | {entity_count} entities removed | {clean_text[:60]}...")

    # ── Deduplicate entities for summary ─────────────────────
    unique_entities = list({e["original"]: e for e in all_entities_removed}.values())

    print(f"\n  ── De-identification Summary ─────────────────")
    print(f"  Total entities removed : {len(all_entities_removed)}")
    print(f"  Unique entities        : {len(unique_entities)}")
    by_type = {}
    for e in all_entities_removed:
        by_type.setdefault(e["type"], 0)
        by_type[e["type"]] += 1
    for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {etype:<25} → {count} removed")

    # ── Build result dict ─────────────────────────────────────
    result = {
        "stage"                        : 5,
        "stage_name"                   : "De-identification of Dataset",
        "interview_id"                 : interview_id,
        "audio_filename"               : stage4_result["audio_filename"],
        "processed_at"                 : now_str(),
        "deidentified_english_full"    : deidentified_full,
        "entities_removed_count"       : len(all_entities_removed),
        "entities_removed_by_type"     : by_type,
        "unique_entities_removed"      : unique_entities,
        # Pass everything forward
        "urdu_full_text"               : stage4_result["urdu_full_text"],
        "verified_urdu_full"           : stage4_result.get("urdu_full_text", ""),
        "english_full_text"            : stage4_result["english_full_text"],
        "verified_english_full"        : stage4_result["verified_english_full"],
        "transcript_quality_score"     : stage4_result["transcript_quality_score"],
        "translation_quality_score"    : stage4_result["translation_quality_score"],
        "duration_minutes"             : stage4_result["duration_minutes"],
        "audio_path"                   : stage4_result["audio_path"],
        "transcript_verification_report"  : stage4_result["transcript_verification_report"],
        "translation_verification_report" : stage4_result["translation_verification_report"],
        "deidentified_segments"        : deidentified_segments,
    }

    # ── Save output ───────────────────────────────────────────
    out_path = os.path.join(config.STAGE5_DIR, f"{interview_id}_deidentified.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 5 complete.")
    return result
