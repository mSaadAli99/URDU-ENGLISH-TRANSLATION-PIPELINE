# Urdu Interview Processing Pipeline

A complete end-to-end pipeline for processing Urdu audio interviews into de-identified English qualitative datasets.

---

## Pipeline Stages

```
Raw Urdu Audio
      ↓
Stage 1 → Urdu Transcription        (whisper-large-v3-turbo)
      ↓
Stage 2 → Verify Urdu Transcript    (confidence scoring)
      ↓
Stage 3 → Urdu → English            (facebook/nllb-200-1.3B)
      ↓
Stage 4 → Verify English            (length-ratio checks)
      ↓
Stage 5 → De-identification         (Microsoft Presidio + spaCy)
      ↓
Stage 6 → Final Export              (JSON + DOCX)
```

---

## Quick Start — Google Colab (Recommended)

1. Open `notebooks/colab_pipeline.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run cells one by one — Cell 4 downloads a sample audio from YouTube automatically
4. Download the zipped outputs at the end

---

## Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 3. (Optional) Download sample audio
python scripts/download_audio.py

# 4. Run pipeline — device is auto-detected (GPU if available, CPU otherwise)
python main.py audio/your_interview.mp3

# 5. Resume from a specific stage after interruption
python main.py audio/your_interview.mp3 --start-stage 3
```

> **CPU note:** On CPU the pipeline will use `int8` quantisation automatically (no manual config edit needed). Expect ~5–10× slower transcription vs GPU.

---

## Folder Structure

```
urdu-pipeline/
├── audio/                         ← Put your audio files here
├── outputs/
│   ├── 1_urdu_transcripts/        ← Stage 1 output (JSON)
│   ├── 2_verified_transcripts/    ← Stage 2 output (JSON)
│   ├── 3_english_translations/    ← Stage 3 output (JSON)
│   ├── 4_verified_translations/   ← Stage 4 output (JSON)
│   ├── 5_deidentified/            ← Stage 5 output (JSON)
│   └── 6_final_dataset/           ← Stage 6: final JSON + DOCX
├── pipeline/
│   ├── transcribe.py              ← Stage 1
│   ├── verify_transcript.py       ← Stage 2
│   ├── translate.py               ← Stage 3
│   ├── verify_translation.py      ← Stage 4
│   ├── deidentify.py              ← Stage 5
│   ├── export.py                  ← Stage 6
│   └── utils.py                   ← Shared helpers
├── scripts/
│   └── download_audio.py          ← Download sample audio from YouTube
├── notebooks/
│   └── colab_pipeline.ipynb       ← Google Colab notebook
├── main.py                        ← Run full pipeline
├── config.py                      ← All settings
└── requirements.txt
```

---

## Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | ASR model size |
| `WHISPER_DEVICE` | auto-detected | `cuda` if GPU found, else `cpu` |
| `WHISPER_COMPUTE_TYPE` | auto-detected | `float16` on GPU, `int8` on CPU |
| `TRANSLATION_MODEL` | `facebook/nllb-200-1.3B` | Translation model |
| `CONFIDENCE_THRESHOLD` | `0.55` | Flag transcript segments below this (calibrated; 0.55 is realistic for conversational speech) |
| `CHUNK_SIZE` | `500` | Max chars per translation chunk |
| `BATCH_TRANSLATION_SIZE` | `5` | Chunks per NLLB batch call |

---

## Models Downloaded Automatically

| Stage | Model | Size |
|---|---|---|
| ASR | whisper-large-v3-turbo | ~800 MB |
| Translation | facebook/nllb-200-1.3B | ~2.5 GB |
| De-ID | spaCy en_core_web_lg | ~560 MB |

---

## Final Output

**JSON** — `outputs/6_final_dataset/{id}_final_dataset.json`
- All text at each pipeline stage
- Quality scores (transcript + translation)
- Full entity removal list
- Flagged segments from both verification stages
- Segment-level data with timestamps

**DOCX** — `outputs/6_final_dataset/{id}_final_dataset.docx`
- Title page
- Interview metadata table
- Verified Urdu transcript
- English translation
- De-identified English text
- Verification reports
