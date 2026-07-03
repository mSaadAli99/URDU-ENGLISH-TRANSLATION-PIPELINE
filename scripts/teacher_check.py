"""Teacher-level audit of the final pipeline output."""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(ROOT, "outputs", "6_final_dataset", "test_audio_final_dataset.json")

with open(path, encoding="utf-8") as f:
    d = json.load(f)

segs = d["segments"]
pii_names = ["Majda", "Zainab", "Kazmi", "Mahajda", "Marjita"]

print("=" * 55)
print("  TEACHER AUDIT — FINAL DATASET")
print("=" * 55)

# 1. PII leakage check
print("\n[1] PII LEAKAGE CHECK")
leaks = []
for s in segs:
    text = s.get("deidentified_english", "")
    for name in pii_names:
        if name.lower() in text.lower():
            leaks.append((s["segment_id"], name, text[:80]))
if leaks:
    for seg_id, name, text in leaks:
        print(f"  LEAK  seg {seg_id:03d}: name={name!r} | {text}")
else:
    print("  PASS — No PII names found in any de-identified segment.")

# 2. False positive [DATE] check
print("\n[2] FALSE POSITIVE [DATE] CHECK")
false_dates = [s for s in segs if "[DATE]" in s.get("deidentified_english", "")]
if false_dates:
    for s in false_dates:
        print(f"  WARN  seg {s['segment_id']:03d}: {s['deidentified_english'][:80]}")
else:
    print("  PASS — No [DATE] tags in any segment.")

# 3. Transcript quality
print("\n[3] QUALITY SCORES")
print(f"  Transcript quality : {d['transcript_quality_score']}/100")
print(f"  Translation quality: {d['translation_quality_score']}/100")
tr = d["transcript_verification_report"]
print(f"  Avg conf (cal)     : {tr.get('avg_calibrated_conf')}")
print(f"  Avg text quality   : {tr.get('avg_text_quality')}")
print(f"  Good segments      : {tr.get('good_segments')}/{tr.get('total_segments')}")
print(f"  Loops removed      : {tr.get('loops_removed')}")

# 4. JSON key names
print("\n[4] JSON KEY NAMES")
expected_keys = [
    "original_transcript", "verified_original_transcript",
    "english_translation", "verified_english_translation",
    "deidentified_english", "entities_removed",
    "transcript_verification_report", "translation_verification_report",
    "segments",
]
for k in expected_keys:
    status = "PASS" if k in d else "FAIL MISSING"
    print(f"  {status:12} {k!r}")
bad_keys = [k for k in ["urdu_transcript", "verified_urdu_transcript"] if k in d]
if bad_keys:
    print(f"  WARN — stale keys still present: {bad_keys}")
else:
    print("  PASS — No stale 'urdu_transcript' keys.")

# 5. Entities summary
print("\n[5] ENTITIES REMOVED")
print(f"  Total       : {d['entities_removed_count']}")
print(f"  By type     : {d['entities_removed_by_type']}")
for e in d["entities_removed"]:
    print(f"    {e['original']!r:20} -> {e['replaced_with']}  ({e['type']})")

# 6. Flagged segments
print("\n[6] FLAGGED SEGMENTS")
print(f"  Count: {len(d['flagged_segments'])}")
for fs in d["flagged_segments"]:
    print(f"  seg {fs.get('segment_id','?'):03d} | {fs.get('reason','?')[:60]}")

print("\n" + "=" * 55)
print("  AUDIT COMPLETE")
print("=" * 55)
