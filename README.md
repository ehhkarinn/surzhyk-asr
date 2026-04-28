# surzhyk-asr
Repository for automatic speech recognition for a Ukrainian variety 

# Overview
Automatic Speech Recognition systems are trained on standardised language varieties, leaving mixed and non-standard speech severely underserved. This project addresses that gap for Surzhyk, which is a Ukrainian-Russian mixed variety with growing sociolinguistic relevance following Russia's full-scale invasion of Ukraine in 2022.
The core contribution is a full ASR pipeline that combines forced alignment, phonological corpus annotation, and fine-tuning of OpenAI's Whisper on Surzhyk-specific data. The result is a dramatic improvement in transcription accuracy for both seen and completely unseen speakers.

# Key Results
| Condition | Baseline Whisper WER | Fine-Tuned Whisper WER |
|---|---|---|
| In-domain (seen speakers) | 34.53% | **4.35%** |
| Out-of-domain (unseen speakers) | 92.96% | **2.82%** |

 The baseline model exhibited two failure modes on Surzhyk: **repetition collapse** (looping the same word) and **premature termination** (stopping after 1–2 words). The fine-tuned model produces clean, Surzhyk-consistent transcriptions across all speakers, including those never seen during training, suggesting the model learned genuine Surzhyk phonological patterns.

# Pipeline
```
Audio Input (16kHz mono WAV)
        ↓
Montreal Forced Aligner (MFA)
  └── Custom Surzhyk pronunciation dictionary
        ↓
Phonological Annotation (Praat TextGrids)
  └── WORD_ORIGIN | PRON_TYPE | FEATURE
      (VOW_SHIFT, CONS_SHIFT, MORPHO_MIX, STRESS_SHIFT, ORTHO_ADAPT)
        ↓
Acoustic Feature Extraction
  └── MFCCs, Pitch (F0), F1/F2 Formants
      Tools: parselmouth, librosa
        ↓
Whisper Fine-Tuning
  └── Base: openai/whisper-small
      10 epochs | 84 training chunks (~10s each)
      Training: 9 speakers | Evaluation: 3 unseen speakers
        ↓
Transcription Output
```

# Repository Structure

| File | Description |
|---|---|
| `preparation.py` | Audio preprocessing and dataset preparation |
| `finetuning.py` / `finetune_combined.py` | Whisper fine-tuning scripts |
| `extract.py` / `extract_embeddings.py` | Acoustic feature extraction |
| `evaluate_acoustic.py` | Acoustic feature evaluation |
| `compare_models.py` / `compare_new.py` | WER comparison between baseline and fine-tuned models |
| `compare_v3_seen.py` / `compare_v3_unseen.py` | In-domain vs out-of-domain evaluation |
| `aggregate_features.py` | Feature aggregation across speakers |
| `check_speaker.py` | Speaker-level data validation |


# Corpus
- **Pilot:** 12 speakers (11 female, 1 male), age 20–25, across Ukraine (predominantly East)
- **Target:** 100 speakers
- **Recording content:** Word list (36 Surzhyk tokens), 5 sentences, connected text (~120 words)
- **Average duration:** 1.5–2.5 minutes per speaker
- **Childhood language profile:** Russian (7), Ukrainian (3), Surzhyk (1), Both (1)


# Tools & Technologies
- Python, PyTorch
- [openai/whisper](https://github.com/openai/whisper)
- [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/)
- Praat / parselmouth
- librosa
- Azure Custom Speech (planned for full corpus scaling)


