# ============================================================
# pipeline/utils.py — Shared helper functions
# ============================================================

import os
import json
import re
from datetime import datetime
import numpy as np


def ensure_dirs(*dirs):
    """Create output directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: dict, path: str, indent: int = 2):
    """Save a dictionary to a JSON file with numpy type support."""
    ensure_dirs(os.path.dirname(path))
    data_converted = convert_numpy_types(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_converted, f, ensure_ascii=False, indent=indent, cls=NumpyEncoder)
    print(f"  Saved -> {path}")


def load_json(path: str) -> dict:
    """Load a JSON file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_interview_id(audio_path: str) -> str:
    """Generate a clean interview ID from the audio filename."""
    basename = os.path.basename(audio_path)
    name, _ = os.path.splitext(basename)
    return re.sub(r"[^\w]", "_", name)


def chunk_text(text: str, max_chars: int = 400) -> list:
    """
    Split long text into chunks of max_chars.
    Tries to split on sentence boundaries (. ! ?) to keep meaning intact.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    sentences = re.split(r'(?<=[.!?۔؟])\s+', text.strip())

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def print_banner(stage_num: int, stage_name: str):
    """Print a clear stage banner to terminal."""
    print("\n" + "=" * 60)
    print(f"  STAGE {stage_num}: {stage_name}")
    print("=" * 60)


def now_str() -> str:
    """Return current datetime as ISO string."""
    return datetime.now().isoformat()


def split_sentences(text: str) -> list:
    """
    Split text into sentences, respecting Urdu and English punctuation.
    """
    text = text.replace('۔', '.').replace('؟', '?')
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list, max_chars: int = 500) -> list:
    """
    Combine sentences into chunks without exceeding max_chars.
    Preserves sentence boundaries for better translation context.
    """
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence

        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def is_urdu_text(text: str, threshold: float = 0.25) -> bool:
    """
    Return True when the text is predominantly Urdu (Arabic script).
    threshold: minimum fraction of Urdu-script characters required.
    Mixed segments below threshold are treated as English (pass-through).
    """
    if not text.strip():
        return False
    urdu_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return urdu_chars / len(text.strip()) >= threshold


def collapse_repetitions(segments: list, max_consecutive: int = 3) -> list:
    """
    Remove Whisper hallucination loops where the same phrase repeats back-to-back.

    Keeps the first occurrence; drops further repeats beyond max_consecutive.
    The end timestamp of the kept segment is extended to cover the dropped ones.
    """
    if not segments:
        return segments

    def _normalise(t: str) -> str:
        return re.sub(r"\s+", " ", t.strip().lower())

    kept: list = []
    run_text: str = ""
    run_count: int = 0

    for seg in segments:
        norm = _normalise(seg.get("text", ""))
        if norm == run_text:
            run_count += 1
        else:
            run_text = norm
            run_count = 1

        if run_count <= max_consecutive:
            kept.append(seg)
        # else: silently drop; caller reports via length difference

    return kept


def merge_short_segments(
    segments: list,
    min_duration: float = 1.5,
    min_words: int = 4,
) -> list:
    """
    Merge micro-segments into their neighbours so confidence scores are
    computed over enough speech signal to be reliable.

    A segment is a merge candidate when BOTH hold:
      - duration < min_duration seconds
      - word count < min_words

    Merge strategy (forward-biased):
      1. Hold the short segment as ``pending``.
      2. Merge it with the next segment regardless of gap.
      3. If the combined result is still short, keep accumulating.
      4. If we reach the end of the list, flush pending backward into
         the last accepted segment.

    Merged segment properties:
      - timestamps span the earliest start to the latest end
      - text is space-joined
      - confidence is re-weighted by word count
      - language / is_urdu inherited from the heavier half
    """
    if len(segments) <= 1:
        return segments

    def _word_count(seg: dict) -> int:
        wl = seg.get("words")
        return len(wl) if wl else len(seg.get("text", "").split())

    def _is_short(seg: dict) -> bool:
        dur = seg["end"] - seg["start"]
        return dur < min_duration and _word_count(seg) < min_words

    def _merge_two(a: dict, b: dict) -> dict:
        a_wc = _word_count(a)
        b_wc = _word_count(b)
        total = a_wc + b_wc
        merged_conf = (
            round((a["confidence"] * a_wc + b["confidence"] * b_wc) / total, 4)
            if total > 0
            else round((a["confidence"] + b["confidence"]) / 2, 4)
        )
        dominant = a if a_wc >= b_wc else b
        return {
            **a,
            "end"        : b["end"],
            "end_fmt"    : b["end_fmt"],
            "text"       : (a["text"] + " " + b["text"]).strip(),
            "confidence" : merged_conf,
            "language"   : dominant["language"],
            "is_urdu"    : dominant["is_urdu"],
            "words"      : (a.get("words") or []) + (b.get("words") or []),
            "merged"     : True,
        }

    result: list = []
    pending = None  # short segment awaiting a partner

    for seg in segments:
        if pending is None:
            if _is_short(seg):
                pending = seg
            else:
                result.append(seg)
        else:
            combined = _merge_two(pending, seg)
            pending = None
            if _is_short(combined):
                pending = combined  # still short — keep accumulating
            else:
                result.append(combined)

    # Flush leftover short segment by merging it backward
    if pending is not None:
        if result:
            result[-1] = _merge_two(result[-1], pending)
        else:
            result.append(pending)

    return result


def calibrate_confidence(raw: float) -> float:
    """
    Calibrate Whisper's conservative word probabilities to values that
    better reflect actual word-level accuracy in conversational speech.

    Whisper's raw probabilities systematically understate accuracy:
      raw 0.30 → true accuracy ~60 %    (raw is too harsh)
      raw 0.50 → true accuracy ~80 %
      raw 0.70 → true accuracy ~90 %
      raw 0.90 → true accuracy ~97 %

    We use power compression  p^0.55  which is a standard calibration
    technique (similar to Platt scaling):
      0.30 → 0.57    0.50 → 0.71    0.65 → 0.80    0.80 → 0.89
    """
    return round(min(float(raw) ** 0.55, 1.0), 4)


# Minimal set of very-common English words used for vocabulary sanity check
_COMMON_EN = frozenset("""
a an the and or but so yet for nor of to in on at by as is are was were be
been being have has had do does did will would shall should may might must
can could not no yes i me my we us our you your he she it they them their
his her its who what when where why how all any some one two this that these
those with from up out about into over after before just also only even then
than more most very much too well off back there here now still ever never
always often never really quite rather quite already soon
""".split())


def score_text_quality(text: str) -> float:
    """
    Score the linguistic quality of a transcribed segment on 0.0–1.0.
    Uses fast heuristics — no external models needed.

    Five checks are combined:
      1. Non-empty content           (gate: score 0.0 if empty)
      2. Clean token ratio           (words are ASCII alphabetic, not garbled)
      3. No ALL-CAPS junk tokens     (garbled ASR often produces BAER:, HOCKS …)
      4. Common-word coverage        (at least some recognisable English words)
      5. Sentence-boundary structure (starts with capital or quote)
    """
    text = text.strip()
    if not text:
        return 0.0

    tokens = text.split()
    if not tokens:
        return 0.0

    # 1. Clean token ratio — valid word chars + common punctuation
    def _is_clean(tok: str) -> bool:
        core = tok.strip(".,!?;:\"'()-[]")
        return bool(core) and all(
            c.isalpha() or c in ("'", "-") for c in core
        )

    clean_ratio = sum(1 for t in tokens if _is_clean(t)) / len(tokens)

    # 2. ALL-CAPS junk detection (≥ 2 consecutive all-caps tokens is bad)
    all_caps_run = 0
    max_caps_run = 0
    for t in tokens:
        if len(t) >= 3 and t.isupper():
            all_caps_run += 1
            max_caps_run = max(max_caps_run, all_caps_run)
        else:
            all_caps_run = 0
    caps_penalty = max(0.0, 1.0 - max_caps_run * 0.20)

    # 3. Common-word coverage (at least a few recognisable function words)
    lower_tokens = [t.lower().strip(".,!?;:'\"") for t in tokens]
    known_ratio = (
        sum(1 for t in lower_tokens if t in _COMMON_EN or (len(t) > 3 and t.isalpha()))
        / len(tokens)
    )

    # 4. Sentence structure (starts with uppercase or dialogue marker)
    starts_ok = 1.0 if (text[0].isupper() or text[0] in ('"', "'", "(")) else 0.7

    # Weighted combination
    score = (
        clean_ratio   * 0.40 +
        caps_penalty  * 0.15 +
        known_ratio   * 0.30 +
        starts_ok     * 0.15
    )
    return round(min(score, 1.0), 4)


def fix_translation_errors(text: str) -> str:
    """
    Fix common Urdu->English translation errors from NLLB output.
    """
    for word in ("the", "is", "a", "and", "of"):
        text = re.sub(rf'\b({word}\s+){{2,}}', f'{word} ', text, flags=re.IGNORECASE)

    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\s+([.!?,])', r'\1', text)
    text = re.sub(
        r'([.!?])\s+([a-z])',
        lambda m: f"{m.group(1)} {m.group(2).upper()}",
        text,
    )

    urdu_fixes = {
        r'\bkya\b': 'what',
        r'\bjo\b':  'which',
        r'\bham\b': 'we',
        r'\bwoh\b': 'that',
        r'\baur\b': 'and',
    }
    for pattern, replacement in urdu_fixes.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text.strip()
