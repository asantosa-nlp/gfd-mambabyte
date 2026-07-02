# Rescoring baseline for R2Q2b

This directory contains the public-release, reproducible version of the
**post-hoc N-best rescoring baseline** requested by reviewer comment R2Q2b.
It is a fair comparison against GFD: the baseline uses the same adapted
MambaByte checkpoint, the same per-language fusion weights, the same byte
limits and heuristic filters, the same 460-utterance test set, the same
normalization, and the same permutation-test protocol. The only difference is
**when** the LM score is applied: after Whisper has produced a fixed N-best
list rather than during search.

## Fairness statement

- LM checkpoint: step-20k adapted MambaByte for each language
- Fusion weights: Javanese λ=0.70, α=0.20; Sundanese λ=0.80, α=0.20
- Constraints: min_output_bytes=30/15, max_output_bytes=160, repetition and
  long-word filters identical to the GFD decoder
- Test set: the same 460-utterance speaker-independent test set
- Normalization: NFC → lowercase → `[^\w\s]` removal → whitespace collapse
- Significance test: matched-pairs permutation test, seed 42, 10,000 permutations

## Pipeline

1. `run_whisper_nbest.sh` / `generate_whisper_nbest.py`
   - Generates a 10-best Whisper list for each utterance in the 460-item test set.
   - The corrected default generation settings are **sampling-based**:
     `do_sample=True`, `temperature=0.7`, `top_p=0.9`, `num_beams=1`,
     `num_return_sequences=10`, `seed=42`.
   - The generated JSONL files are **not** committed to git.
2. `run_rescore_whisper_nbest.sh` / `rescore_whisper_nbest.py`
   - Scores each hypothesis with the adapted MambaByte checkpoint.
   - Applies the exact GFD fusion formula to whole hypotheses.
   - Computes the N-best oracle from the same 10-best list.
3. `run_build_rescoring_report.sh` / `build_rescoring_report.py`
   - Renders the human-readable report files from the saved summary JSONs.

## Results

| System | JV WER | JV CER | SU WER | SU CER |
|---|---:|---:|---:|---:|
| Zero-shot Whisper (1-best) | 58.16 | 14.44 | 61.64 | 12.70 |
| N-best rescoring (adapted LM, post-hoc, same λ/α) | 60.53 | 15.47 | 63.21 | 13.30 |
| N-best oracle (lowest-WER in 10-best) | 47.72 | 11.75 | 52.29 | 11.39 |
| GFD first-pass fusion (our method) | 51.87 | 15.62 | 54.72 | 13.33 |

One-line finding: **GFD outperforms post-hoc rescoring in both languages**,
but the true 10-best oracle is better than GFD in both languages. The earlier
oracle=rescoring equality was a beam-collapse artifact from a broken N-best pool,
not the genuine result.

## Prerequisites

The rescoring step consumes two kinds of first-pass prediction JSONs produced by
the main pipeline:

- zero-shot Whisper predictions:
  - `results/baseline_large_jv_lnews/predictions.json`
  - `results/baseline_large_su_lnews/predictions.json`
- first-pass GFD predictions:
  - `results/mb_jv_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20/predictions.json`
  - `results/gs20k_su_lm80_ag20_blind/predictions.json`

For convenience in this release package, copies of those files are shipped in
`experiments/rescoring/baseline_predictions/` under clear names:

- `results/baseline_large_jv_lnews/predictions.json`
  → `experiments/rescoring/baseline_predictions/jv_zero_shot_predictions.json`
- `results/baseline_large_su_lnews/predictions.json`
  → `experiments/rescoring/baseline_predictions/su_zero_shot_predictions.json`
- `results/mb_jv_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20/predictions.json`
  → `experiments/rescoring/baseline_predictions/jv_gfd_predictions.json`
- `results/gs20k_su_lm80_ag20_blind/predictions.json`
  → `experiments/rescoring/baseline_predictions/su_gfd_predictions.json`

The release rescoring script reads the shipped copies directly, so the
self-contained verification mode works without any `results/...` mirror paths.

## Reproduce

The fixed parameters are collected in [`configs/rescoring.yaml`](../../configs/rescoring.yaml).

```bash
# 1) Generate the Whisper 10-best lists with the corrected sampling defaults
bash experiments/rescoring/run_whisper_nbest.sh --cuda 0 --lang jv \
  --out-dir experiments/rescoring/nbest/jv_t07_topp09_s42
bash experiments/rescoring/run_whisper_nbest.sh --cuda 1 --lang su \
  --out-dir experiments/rescoring/nbest/su_t07_topp09_s42

# 2) Rescore the fixed N-best lists
bash experiments/rescoring/run_rescore_whisper_nbest.sh --cuda 0 --lang jv \
  --nbest-file experiments/rescoring/nbest/jv_t07_topp09_s42/jv_nbest.jsonl
bash experiments/rescoring/run_rescore_whisper_nbest.sh --cuda 1 --lang su \
  --nbest-file experiments/rescoring/nbest/su_t07_topp09_s42/su_nbest.jsonl

# 3) Render RESULTS.md and FINDINGS.md
bash experiments/rescoring/run_build_rescoring_report.sh
```

The markdown report files are written to `experiments/rescoring/RESULTS.md`
and `experiments/rescoring/FINDINGS.md`. The summary JSONs are stored in
`experiments/rescoring/summaries/`.

## Running modes and prerequisites

The rescoring baseline can be run in two modes, with different prerequisites. Choose the one that
matches what you want to verify.

### Mode 1 — Rescore from shipped predictions (self-contained; recommended for verification)

This reproduces the rescoring result (Table: zero-shot / rescoring / oracle / GFD WER, and the
GFD-vs-rescoring permutation test) **using only files inside this repository**. It does not require
the speech audio or the model checkpoints, because the first-pass zero-shot and GFD predictions are
shipped under `experiments/rescoring/baseline_predictions/`:

| Role | Shipped file |
|---|---|
| Javanese zero-shot predictions | `experiments/rescoring/baseline_predictions/jv_zero_shot_predictions.json` |
| Sundanese zero-shot predictions | `experiments/rescoring/baseline_predictions/su_zero_shot_predictions.json` |
| Javanese GFD predictions | `experiments/rescoring/baseline_predictions/jv_gfd_predictions.json` |
| Sundanese GFD predictions | `experiments/rescoring/baseline_predictions/su_gfd_predictions.json` |

Invoke the rescore-and-report step pointing explicitly at the shipped prediction files (and the
shipped N-best files if you have regenerated them; see Mode 2). The script reads these shipped
copies directly — no `data/` manifests or `checkpoints/` are needed for this mode.

> If you use the helper `run_rescore_whisper_nbest.sh`, pass the N-best file explicitly with
> `--nbest-file ...`; the default search path is `experiments/rescoring/nbest/`, which is gitignored
> and not shipped (see Mode 2 for regeneration).

### Mode 2 — End-to-end (requires external inputs documented in `docs/DATA.md`)

This regenerates the first-pass N-best lists and predictions from scratch and then rescores. It
requires the external prerequisites that are **not** redistributed in this repository:

- the 460-utterance test manifests `data/speech_corpus/local_jv/test.jsonl` and
  `data/speech_corpus/local_su/test.jsonl`;
- the step-20k adapted MambaByte checkpoints
  `checkpoints/mb_cpt_jv_v5_clean/step_0020000.pt` and
  `checkpoints/mb_cpt_su_v5_clean/step_0020000.pt`.

See `docs/DATA.md` for where to obtain these and where to place them. Once they are in place, run the
N-best generation step (`generate_whisper_nbest.py`) followed by the rescore-and-report step.

### Note on `PYTHONPATH`

The rescoring script imports `gfd.lm_adapters`, which resolves only when `src/decoding` is on
`PYTHONPATH` (the package root `src/` alone is not sufficient). Set it as:

```
export PYTHONPATH="$(pwd)/src/decoding:$(pwd)/src"
```

before invoking the rescoring scripts (the shell wrappers set this for you).
