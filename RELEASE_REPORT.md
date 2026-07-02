# Release report

This report captures the assembled public-release directory and the verification performed during assembly.

## Full file tree

```text
CITATION.cff
EXTRACTION_LOG.md
LICENSE
README.md
RELEASE_REPORT.md
configs/cpt_javanese.yaml
configs/cpt_jv_v5_clean.yaml
configs/cpt_su_v5_clean.yaml
configs/cpt_sundanese.yaml
configs/decode_javanese.yaml
configs/decode_sundanese.yaml
configs/mb_jv_nocpt_blind_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5cleanS10k_ckptsel1k.yaml
configs/mb_jv_zs_large_v5cleanS10k_no_gate_lnews_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5cleanS20k_ckptsel1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag15.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag15_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag20.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag20_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag25.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm60_ag25_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag15.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag15_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag20.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag20_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag25.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm70_ag25_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag15.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag15_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag20.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag20_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag25.yaml
configs/mb_jv_zs_large_v5cleanS20k_gs_lm80_ag25_1k.yaml
configs/mb_jv_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5cleanS30k_ckptsel1k.yaml
configs/mb_jv_zs_large_v5cleanS30k_no_gate_lnews_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5cleanS40k_no_gate_lnews_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5clean_ckptsel1k.yaml
configs/mb_jv_zs_large_v5clean_gate_lnews_lm70_ag0.20.yaml
configs/mb_jv_zs_large_v5clean_gs_lm60_ag15.yaml
configs/mb_jv_zs_large_v5clean_gs_lm60_ag20.yaml
configs/mb_jv_zs_large_v5clean_gs_lm60_ag25.yaml
configs/mb_jv_zs_large_v5clean_gs_lm70_ag15.yaml
configs/mb_jv_zs_large_v5clean_gs_lm70_ag20.yaml
configs/mb_jv_zs_large_v5clean_gs_lm70_ag25.yaml
configs/mb_jv_zs_large_v5clean_gs_lm80_ag15.yaml
configs/mb_jv_zs_large_v5clean_gs_lm80_ag20.yaml
configs/mb_jv_zs_large_v5clean_gs_lm80_ag25.yaml
configs/mb_jv_zs_large_v5clean_no_gate_lnews_lm70_ag0.20.yaml
configs/mb_su_nocpt_blind_lm80_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5cleanS10k_ckptsel1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS10k_no_gate_lnews_lm70_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_ckptsel1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag15_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag15_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag20_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag20_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag25_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm60_ag25_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag15_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag15_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag20_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag20_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag25_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm70_ag25_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag15_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag15_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag20_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag20_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag25_1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag25_minb15.yaml
configs/mb_su_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5cleanS30k_ckptsel1k_minb15.yaml
configs/mb_su_zs_large_v5cleanS30k_no_gate_lnews_lm70_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5cleanS40k_no_gate_lnews_lm70_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5clean_ckptsel1k_minb15.yaml
configs/mb_su_zs_large_v5clean_gate_lnews_lm70_ag0.20_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm60_ag15_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm60_ag20_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm60_ag25_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm70_ag15_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm70_ag20_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm70_ag25_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm80_ag15_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm80_ag20_minb15.yaml
configs/mb_su_zs_large_v5clean_gs_lm80_ag25_minb15.yaml
configs/mb_su_zs_large_v5clean_no_gate_lnews_lm70_ag0.20_minb15.yaml
configs/seeds.yaml
data_splits/javanese_search_1000.txt
data_splits/javanese_test_460.txt
data_splits/sundanese_search_1000.txt
data_splits/sundanese_test_460.txt
docs/CONFIG_REFERENCE.md
docs/DATA.md
docs/REPRODUCE.md
logs/hparam_search/jv_gridsearch.log
logs/hparam_search/su_gridsearch.log
requirements.txt
results/checkpoint_used.md
src/cpt/README.md
src/cpt/__init__.py
src/cpt/prepare_cpt_v2.py
src/cpt/train.py
src/decoding/README.md
src/decoding/gfd/__init__.py
src/decoding/gfd/asr_adapters.py
src/decoding/gfd/byte_gfd.py
src/decoding/gfd/entropy_gate.py
src/decoding/gfd/lattice.py
src/decoding/gfd/lm_adapters.py
src/scoring/README.md
src/scoring/run_exp_mb.py
src/scoring/run_whisper_baseline.py
src/search/README.md
src/search/gen_gs20k_1k_configs.py
src/search/launch_gs20k_1k_remote.sh
src/search/make_search_sample.py
src/search/sample_train_manifest.py
src/selection/README.md
src/selection/_ckptsel1k_jv_10k.sh
src/selection/_ckptsel1k_su_10k.sh
src/selection/launch_ckptsel1k.sh
src/whisper/README.md
src/whisper/asr_adapters.py
```

## Step 0 inventory final status

| Item | Status | Location / note |
|---|---|---|
| GFD decoding loop / beam search | FOUND | `src/decoding/gfd/byte_gfd.py` |
| Prefix-trie cache implementation | FOUND | `src/decoding/gfd/lm_adapters.py` |
| Whisper inference wrapper + decode settings | FOUND | `src/whisper/asr_adapters.py`, `configs/decode_*.yaml` |
| LoRA continual-pre-training (CPT) script | FOUND | `src/cpt/train.py` |
| Checkpoint-selection script | FOUND | `src/selection/launch_ckptsel1k.sh`, `_ckptsel1k_*.sh` |
| WER normalization + scoring script | FOUND | `src/scoring/run_exp_mb.py` |
| Hyperparameter grid-search driver | FOUND | `src/search/gen_gs20k_1k_configs.py`, `src/search/launch_gs20k_1k_remote.sh` |
| Hyperparameter grid-search output logs | FOUND | `logs/hparam_search/jv_gridsearch.log`, `logs/hparam_search/su_gridsearch.log` |
| Random-seed definitions | FOUND | `configs/seeds.yaml` |
| Search/test split lists | FOUND | `data_splits/*.txt` |
| Config files | FOUND | `configs/*.yaml` |
| Per-language LoRA adapter weights | FOUND in source, NOT COPIED | `checkpoints/mb_cpt_jv_v5_clean/step_0020000.pt` (128M), `checkpoints/mb_cpt_su_v5_clean/step_0020000.pt` (128M) |
| Environment / dependency definitions | PARTIAL | `requirements.txt` found; `environment.yml`, `pyproject.toml`, lockfiles not present in this checkout |

## R1Q8 checklist mapping

| Item | Status | Location / note |
|---|---|---|
| GFD decoding code | `src/decoding/gfd/byte_gfd.py`, `src/decoding/gfd/lattice.py`, `src/decoding/gfd/entropy_gate.py` |
| prefix-trie cache implementation | `src/decoding/gfd/lm_adapters.py` |
| exact Whisper decoding settings | `src/whisper/asr_adapters.py`, `configs/decode_javanese.yaml`, `configs/decode_sundanese.yaml` |
| LoRA / CPT scripts | `src/cpt/train.py`, `src/cpt/prepare_cpt_v2.py`, `configs/cpt_*.yaml` |
| checkpoint-selection scripts | `src/selection/launch_ckptsel1k.sh`, `src/selection/_ckptsel1k_jv_10k.sh`, `src/selection/_ckptsel1k_su_10k.sh` |
| WER normalization / scoring scripts | `src/scoring/run_exp_mb.py`, `src/scoring/run_whisper_baseline.py` |
| hyperparameter-search logs | `logs/hparam_search/jv_gridsearch.log`, `logs/hparam_search/su_gridsearch.log` |
| random seeds (incl. 42) | `configs/seeds.yaml` |
| split lists (1,000 search + 460 test, per language) | `data_splits/javanese_search_1000.txt`, `data_splits/javanese_test_460.txt`, `data_splits/sundanese_search_1000.txt`, `data_splits/sundanese_test_460.txt` |
| config files identifying checkpoint 20k | `configs/decode_javanese.yaml`, `configs/decode_sundanese.yaml`, `results/checkpoint_used.md` |
| README with reproduction steps for Table 4 | `README.md`, `docs/REPRODUCE.md` |
| per-language LoRA adapter weights + dataset-prep instructions | weights flagged for manual add; `docs/DATA.md`, `src/cpt/prepare_cpt_v2.py` |

## Verification notes

- Search/test split overlap check: **0** for Javanese and **0** for Sundanese.
- Safety sweep: no secret strings, absolute home paths, internal hostnames, or HF tokens remain in the release tree.
- Safety sweep: no `.wav`, `.flac`, `.mp3`, `.pt`, `.bin`, `.safetensors`, or files >50 MB were copied into the release tree.
- The release intentionally omits raw audio and raw CPT corpora.
- The LoRA adapter checkpoints are present in the source tree only and remain **not copied**; each is 128 MB.
- `__pycache__` artifacts were removed from the release tree after a verification compile step.

## Rescoring experiment integration

```text
experiments/rescoring/README.md
experiments/rescoring/INTEGRATION_REPORT.md
experiments/rescoring/generate_whisper_nbest.py
experiments/rescoring/run_whisper_nbest.sh
experiments/rescoring/rescore_whisper_nbest.py
experiments/rescoring/run_rescore_whisper_nbest.sh
experiments/rescoring/build_rescoring_report.py
experiments/rescoring/run_build_rescoring_report.sh
experiments/rescoring/RESULTS.md
experiments/rescoring/FINDINGS.md
experiments/rescoring/summaries/jv_summary.json
experiments/rescoring/summaries/su_summary.json
configs/rescoring.yaml
```

R1Q8 extension for the rescoring baseline:

| Item | Status | Location / note |
|---|---|---|
| rescoring baseline scripts | FOUND | `experiments/rescoring/{generate_whisper_nbest.py,run_whisper_nbest.sh,rescore_whisper_nbest.py,run_rescore_whisper_nbest.sh,build_rescoring_report.py,run_build_rescoring_report.sh}` |
| rescoring baseline configs | FOUND | `configs/rescoring.yaml` |
| rescoring summaries | FOUND | `experiments/rescoring/summaries/jv_summary.json`, `experiments/rescoring/summaries/su_summary.json` |
| rescoring baseline README | FOUND | `experiments/rescoring/README.md` |
| raw N-best JSONL | GITIGNORED | `experiments/rescoring/nbest/` is ignored via `.gitignore` and regenerable with `run_whisper_nbest.sh` |

## Still requiring your input

- Choose the final license to replace `LICENSE`.
- Fill the DOI/date placeholders in `CITATION.cff`.
- Decide whether to add the two 128 MB LoRA adapter checkpoints to the public release.
- Confirm the verbatim CPT-text leakage statement against the original corpora, since those corpora are absent from this checkout.
