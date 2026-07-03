# ============================================================
# config.py — Central configuration for Urdu Pipeline
# All model names, paths, thresholds in one place
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR       = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

STAGE1_DIR = os.path.join(OUTPUT_DIR, "1_urdu_transcripts")
STAGE2_DIR = os.path.join(OUTPUT_DIR, "2_verified_transcripts")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "3_english_translations")
STAGE4_DIR = os.path.join(OUTPUT_DIR, "4_verified_translations")
STAGE5_DIR = os.path.join(OUTPUT_DIR, "5_deidentified")
STAGE6_DIR = os.path.join(OUTPUT_DIR, "6_final_dataset")

# ── Stage 1: ASR Model ───────────────────────────────────────
WHISPER_MODEL        = "large-v3-turbo"   # openai/whisper-large-v3-turbo via faster-whisper
# None = auto-detect language per segment (correct for code-switched Urdu/English interviews)
# Set to "ur" only if the entire audio is Urdu with no English
WHISPER_LANGUAGE     = None
# Neutral prompt for mixed interviews — does NOT bias toward one language
WHISPER_INITIAL_PROMPT = "This is a research interview. The speakers may use both Urdu and English."
# Temperature fallback list: Whisper retries at higher temperatures when a segment scores poorly.
# 0.0 (greedy) is tried first; if compression_ratio or log_prob is bad, it falls back in order.
WHISPER_TEMPERATURE  = [0.0, 0.2, 0.4, 0.6]
WHISPER_BEAM_SIZE    = 5                   # Higher beam = better quality

# ── Stage 1: Quality filters (Whisper decode thresholds) ─────
# Segments where no-speech probability exceeds this are silently dropped (music, noise, etc.)
WHISPER_NO_SPEECH_THRESHOLD      = 0.60
# If a segment's text compresses to > this ratio it is repetitive / hallucinated — retried
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
# Segments with avg log-probability below this are flagged as unreliable — retried
WHISPER_LOG_PROB_THRESHOLD       = -1.0

# ── Stage 1: Segment merging ──────────────────────────────────
# After raw ASR, micro-segments (< this duration AND < this many words) are merged
# with their neighbour to produce more reliable confidence estimates.
WHISPER_MERGE_MIN_DURATION = 1.5   # seconds — merge segments shorter than this
WHISPER_MERGE_MIN_WORDS    = 4     # words — merge if word count is also below this

# ── Stage 1: Repetition / hallucination filter ────────────────
# Whisper sometimes loops the same phrase; collapse runs longer than this
REPETITION_MAX_CONSECUTIVE = 3   # Flag a segment if same text appeared N+ times in a row

# Auto-detect GPU; fall back to CPU gracefully
try:
    import torch as _torch
    _HAS_GPU = _torch.cuda.is_available()
    if _HAS_GPU:
        WHISPER_DEVICE       = "cuda"
        WHISPER_COMPUTE_TYPE = "float16"   # float16 is fast on GPU
    else:
        WHISPER_DEVICE       = "cpu"
        WHISPER_COMPUTE_TYPE = "int8"      # int8 is the only quantisation faster-whisper supports on CPU
except ImportError:
    _HAS_GPU             = False
    WHISPER_DEVICE       = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

# ── Stage 2: Transcript Verification ─────────────────────────
# 0.55 is realistic for accented conversational speech in Whisper.
# 0.60+ is appropriate for clean read speech; 0.75+ for studio audio.
CONFIDENCE_THRESHOLD = 0.55    # Flag segments below this confidence
MIN_QUALITY_SCORE    = 60      # Minimum acceptable transcript quality (0-100)

# ── Stage 3: Translation Model ───────────────────────────────
TRANSLATION_MODEL    = "facebook/nllb-200-1.3B"  # Best quality — use on Colab GPU
# Lighter fallback for local CPU when RAM is limited (~300 MB vs ~2.5 GB)
CPU_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ur-en"
NLLB_SRC_LANG        = "urd_Arab"   # Urdu source language code for NLLB
NLLB_TGT_LANG        = "eng_Latn"   # English target language code for NLLB
CHUNK_SIZE           = 500           # Max characters per sentence chunk (for sentence-based splitting)
BATCH_TRANSLATION_SIZE = 16 if _HAS_GPU else 5   # Larger batches on GPU; smaller on CPU to avoid OOM
USE_CONTEXT_WINDOW   = True           # Use previous segment for context (better continuity)
CONTEXT_WINDOW_SIZE  = 2              # Look back N segments for context

# ── Stage 4: Translation Verification ────────────────────────
MIN_TRANSLATION_SCORE = 50     # Minimum acceptable translation quality (0-100)

# ── Stage 5: De-identification ───────────────────────────────
SPACY_MODEL = "en_core_web_lg"

# Generic temporal words that Presidio incorrectly flags as DATE_TIME PII.
# These are common English words with no identifying value and must NOT be removed.
# Examples: "today", "night", "a day", "morning", "yesterday", "later".
GENERIC_TEMPORAL_WORDS = frozenset({
    "today", "tomorrow", "yesterday", "now", "then", "soon", "later",
    "morning", "afternoon", "evening", "night", "midnight", "noon",
    "daily", "weekly", "monthly", "yearly", "annual",
    "a day", "one day", "the day", "each day", "every day",
    "a week", "a month", "a year",
    "at night", "at noon", "last night", "last week", "last month",
    "next week", "next month", "next year",
    "this week", "this month", "this year",
    "the morning", "the evening", "the night", "the afternoon",
    "recently", "currently", "always", "never", "sometimes", "often",
})

# Entity type → replacement tag mapping
ENTITY_REPLACEMENTS = {
    "PERSON":       "[NAME]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS":"[EMAIL]",
    "LOCATION":     "[LOCATION]",
    "GPE":          "[LOCATION]",
    "ORG":          "[ORGANIZATION]",
    "NRP":          "[ORGANIZATION]",
    "DATE_TIME":    "[DATE]",
    "US_SSN":       "[ID]",
    "US_PASSPORT":  "[ID]",
    "IBAN_CODE":    "[ID]",
    "CREDIT_CARD":  "[ID]",
    "URL":          "[URL]",
    "IP_ADDRESS":   "[ID]",
}

# ── Stage 6: Export ───────────────────────────────────────────
DOCX_TITLE         = "Urdu Interview — De-identified Qualitative Dataset"
DOCX_AUTHOR        = "Urdu Pipeline"
FINAL_JSON_INDENT  = 2
