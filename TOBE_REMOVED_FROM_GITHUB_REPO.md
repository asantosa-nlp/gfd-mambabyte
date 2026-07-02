# Files that may be removed before publishing the public GitHub repo

This is a curation list, not a mandatory deletion list.
I separated **strong removal candidates** (generated / transient / internal-process files) from
**optional removals** (auxiliary experiments or duplicate snapshots that are not needed for the
main reproduction path documented in `README.md` and `docs/REPRODUCE.md`).

## Strong removal candidates

These files do not help a reviewer reproduce the paper's reported results; they are either build
artifacts, raw logs, or internal process notes.

### 1) Python bytecode / cache artifacts
Remove the files below and the now-empty `__pycache__/` directories:
- `experiments/rescoring/__pycache__/build_rescoring_report.cpython-310.pyc`
- `experiments/rescoring/__pycache__/generate_whisper_nbest.cpython-310.pyc`
- `experiments/rescoring/__pycache__/rescore_whisper_nbest.cpython-310.pyc`
- `src/decoding/gfd/__pycache__/__init__.cpython-310.pyc`
- `src/decoding/gfd/__pycache__/lm_adapters.cpython-310.pyc`

### 2) Raw hparam-search logs
These are run logs, not reproduction assets:
- `logs/hparam_search/jv_gridsearch.log`
- `logs/hparam_search/su_gridsearch.log`

### 3) Internal process / audit reports
These are transparency artifacts from the repo assembly/correction process, not part of the paper
reproduction path:
- `EXTRACTION_LOG.md`
- `RELEASE_REPORT.md`
- `experiments/rescoring/CORRECTIONS.md`
- `experiments/rescoring/CORRECTION_REPORT.md`
- `experiments/rescoring/INTEGRATION_REPORT.md`

## Optional removals if you want a leaner public release

These files are useful for provenance or auxiliary experiments, but they are not needed for the
main documented reproduction path (Table 4 + the corrected rescoring baseline).

### 4) Search / grid-search subsystem
If you do **not** want to ship the curiosity grid-search tooling, the whole subsystem can go:
- `src/search/README.md`
- `src/search/gen_gs20k_1k_configs.py`
- `src/search/launch_gs20k_1k_remote.sh`
- `src/search/make_search_sample.py`
- `src/search/sample_train_manifest.py`

If that subsystem is removed, the generated grid-search configs below are also removable:

#### Javanese grid-search configs (18 files)
- `configs/mb_jv_zs_large_v5clean_gs_lm{60,70,80}_ag{15,20,25}.yaml`
- `configs/mb_jv_zs_large_v5cleanS20k_gs_lm{60,70,80}_ag{15,20,25}.yaml`
- `configs/mb_jv_zs_large_v5cleanS20k_gs_lm{60,70,80}_ag{15,20,25}_1k.yaml`

#### Sundanese grid-search configs (18 files)
- `configs/mb_su_zs_large_v5clean_gs_lm{60,70,80}_ag{15,20,25}_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS20k_gs_lm{60,70,80}_ag{15,20,25}_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS20k_gs_lm{60,70,80}_ag{15,20,25}_1k_minb15.yaml`

Reason: these are curiosity / hyperparameter-grid outputs. The public docs already record the final
chosen settings, so the full grid-search matrix is not needed for the main paper reproduction.
If you ever need the 1k variants again, `src/search/gen_gs20k_1k_configs.py` can regenerate them
from the 300-utterance source configs.

### 5) Duplicate CPT config snapshots
These appear to be redundant aliases of the canonical CPT configs already kept in the repo:
- `configs/cpt_jv_v5_clean.yaml` (canonical equivalent exists as `configs/cpt_javanese.yaml`)
- `configs/cpt_su_v5_clean.yaml` (canonical equivalent exists as `configs/cpt_sundanese.yaml`)

### 6) Auxiliary ablation snapshots
If the goal is only to preserve the final reported Table 4 / rescoring path, these auxiliary
variants are not required by the current documentation:
- `configs/mb_jv_nocpt_blind_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5clean_no_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5clean_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5cleanS10k_no_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5cleanS30k_no_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_jv_zs_large_v5cleanS40k_no_gate_lnews_lm70_ag0.20.yaml`
- `configs/mb_su_nocpt_blind_lm80_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5clean_no_gate_lnews_lm70_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5clean_gate_lnews_lm70_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS10k_no_gate_lnews_lm70_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS20k_no_gate_lnews_lm70_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS30k_no_gate_lnews_lm70_ag0.20_minb15.yaml`
- `configs/mb_su_zs_large_v5cleanS40k_no_gate_lnews_lm70_ag0.20_minb15.yaml`

Reason: these are auxiliary / ablation snapshots. The main release docs use
`configs/decode_javanese.yaml`, `configs/decode_sundanese.yaml`, `configs/rescoring.yaml`, and the
checkpoint-selection configs, so these extra variants are not required for the main reproduction
path.

### 7) Redundant result note
- `results/checkpoint_used.md`

Reason: the same checkpoint-selection statement is already documented in `README.md` and
`docs/REPRODUCE.md`, so this tiny note is duplicative if you want a leaner public tree.

## Files I would **not** remove

To avoid breaking the documented reproduction path, I would keep at least:
- `src/{cpt,decoding,scoring,selection,whisper}/...`
- `configs/{cpt_javanese.yaml,cpt_sundanese.yaml,decode_javanese.yaml,decode_sundanese.yaml,rescoring.yaml,seeds.yaml}`
- the checkpoint-selection configs (`configs/*ckptsel1k*.yaml`)
- `data_splits/`
- `docs/`
- `experiments/rescoring/{README.md,RESULTS.md,FINDINGS.md,build_rescoring_report.py,generate_whisper_nbest.py,rescore_whisper_nbest.py,run_*.sh,summaries/,baseline_predictions/}`

These are the core files needed to reproduce the reported experiments and the corrected rescoring
baseline.
