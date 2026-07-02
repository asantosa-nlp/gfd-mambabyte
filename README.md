# GFD + MambaByte public release

Public-release snapshot for the implementation described in the paper
*Generative Fusion Decoding with MambaByte for Low-Resource Regional Language
Speech Recognition: A Case Study on Javanese and Sundanese*.

> Paper: <citation>, IJIES (under revision).
> DOI: <DOI_TBD>

Headline result in the paper: **Javanese 51.87% WER / Sundanese 54.72% WER**.

## Repository layout

- `src/decoding/` — GFD byte-search decoder and prefix-trie cache
- `src/whisper/` — Whisper inference wrapper and decode settings
- `src/cpt/` — LoRA continual pre-training for MambaByte
- `src/selection/` — checkpoint-selection launch scripts
- `src/scoring/` — WER normalization / scoring runner
- `src/search/` — hyperparameter grid-search config generator
- `configs/` — training / decoding configs and seed registry
- `data_splits/` — 1,000-utt search and 460-utt test lists
- `logs/` — condensed grid-search logs
- `results/` — checkpoint selection note and release notes
- `docs/` — reproduction, data provenance, and config reference

## Quick install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Table 4

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the end-to-end sequence.

Minimal run sketch:

```bash
export PYTHONPATH="$PWD/src/decoding"
python3 src/scoring/run_exp_mb.py configs/decode_javanese.yaml --split blind
python3 src/scoring/run_exp_mb.py configs/decode_sundanese.yaml --split blind
```

## Experiments

- [`experiments/rescoring/README.md`](experiments/rescoring/README.md) — fair post-hoc N-best rescoring baseline for reviewer comment R2Q2b.
- The corrected rescoring baseline result is: **GFD beats post-hoc rescoring in both languages, while the true 10-best oracle remains better than GFD**; see the experiment README for the full table and reproduction commands.

## Notes

- The public release does **not** include raw audio, raw corpora, or large checkpoints.
- The checkpoint behind the Table 4 rows is step 20k (`step_0020000.pt`).
