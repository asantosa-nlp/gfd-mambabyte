# Data and External Prerequisites

This repository contains all **code, configurations, split lists, seeds, and result
summaries** needed to understand and reproduce the experiments. It deliberately does **not**
redistribute the speech audio, the raw text corpora, or the model checkpoints, because these
are large and/or governed by their original licenses. This document specifies exactly what
external inputs are required, where to obtain them, and where to place them so the documented
commands run.

## 1. Speech data (acoustic input and test transcripts)

- **Source:** INDspeech_NEWSTRA_EthnicSR (Javanese and Sundanese read-speech corpus).
- **Obtain from:** https://github.com/s-sakti/data_indsp_news_ethnicsr (License CC BY-NC-SA 4.0); this
  repository does not redistribute the audio.
- **Place at:**
  - `data/speech_corpus/local_jv/` and `data/speech_corpus/local_su/`
  - The manifests the configs expect are `test.jsonl` (the 460-utterance test set) and
    `train_1000_seed42.jsonl` (the 1,000-utterance search set), per language.
- **Which utterances:** the exact utterance IDs that constitute the 1,000-utterance search set
  and the 460-utterance test set per language are shipped in `data_splits/` so the partition is
  fully specified and verifiable even though the audio is not included. The two partitions are
  speaker-disjoint, and the test set is never used during adaptation or hyperparameter search.

## 2. Text corpora (for continual pre-training, CPT)

- **Sources:** CC-100, Wikipedia, and in-domain news text for Javanese and Sundanese.
- **Obtain from:** the original public sources for each; this repository does not redistribute
  the raw corpora.
- **Place at:**
  - `data/text_corpus/combined_jv_v2_clean.txt`
  - `data/text_corpus/combined_su_v2_clean.txt`
- **Preprocessing applied to produce the "clean" corpora** (so others can reproduce the cleaning):
  - empty-line removal,
  - a light English-contamination filter (see `src/cpt/prepare_cpt_v2.py`),
  - a **leak-removal pass** that filters CPT lines against the test and search reference
    transcripts (this is why the released CPT corpora are labelled "de-leaked" in the configs).
  - No generic corpus-internal exact-line or near-duplicate deduplication step is applied; this
    is stated explicitly for transparency.

## 3. Model checkpoints

- **CPT (LoRA) checkpoints:** the continual-pre-trained MambaByte checkpoints, in particular the
  **step-20k** checkpoint used for all reported results.
- **Obtain from:** released as a GitHub release asset / available on request (the checkpoints are
  not committed to the git tree). See the repository's Releases page.
- **Place at:**
  - `checkpoints/mb_cpt_jv_v5_clean/step_0020000.pt`
  - `checkpoints/mb_cpt_su_v5_clean/step_0020000.pt`
- **Base model:** the public `MambaByte_PG19_972M` checkpoint (token-free byte-level SSM), obtained
  from its original distribution.

## 4. Precomputed baseline predictions (only needed for the rescoring baseline)

The rescoring experiment (`experiments/rescoring/`) consumes first-pass prediction files produced
by the main pipeline. See `experiments/rescoring/README.md` for how to generate them; the small
prediction JSONs are also shipped under `experiments/rescoring/baseline_predictions/` so the
rescoring walkthrough is self-contained.

## 5. Leakage / overlap verification

A summary of the CPT-vs-test text-overlap check is shipped at
`docs/leakage_report.md` (and `docs/leakage/` for the per-pair detail), so the
disjointness claim is verifiable from this repository **without** redistributing the raw corpora.
See that file for the method and results.

## Directory layout expected after placing external inputs

```
gfd-mambabyte-release/
├── data/
│   ├── speech_corpus/local_jv/{test.jsonl, train_1000_seed42.jsonl}
│   ├── speech_corpus/local_su/{test.jsonl, train_1000_seed42.jsonl}
│   └── text_corpus/{combined_jv_v2_clean.txt, combined_su_v2_clean.txt}
├── checkpoints/
│   ├── mb_cpt_jv_v5_clean/step_0020000.pt
│   └── mb_cpt_su_v5_clean/step_0020000.pt
└── (code/configs/splits/docs as shipped)
```

Once these inputs are in place, the commands in `docs/REPRODUCE.md` run end to end.
