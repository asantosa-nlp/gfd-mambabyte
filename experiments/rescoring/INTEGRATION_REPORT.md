# Rescoring experiment integration report

This report records how the reviewer-requested N-best rescoring baseline was integrated into the public release repository.

## Final file tree

```text
experiments/rescoring/
├── FINDINGS.md
├── INTEGRATION_REPORT.md
├── README.md
├── RESULTS.md
├── build_rescoring_report.py
├── generate_whisper_nbest.py
├── rescore_whisper_nbest.py
├── run_build_rescoring_report.sh
├── run_rescore_whisper_nbest.sh
├── run_whisper_nbest.sh
└── summaries/
    ├── jv_summary.json
    └── su_summary.json
```

## Config added

```text
configs/rescoring.yaml
```

## Import / path repointing map

- `run_whisper_nbest.sh`
  - repo-root detection changed to the release repo root
  - output directories now default to `experiments/rescoring/nbest/`
  - `PYTHONPATH` now includes `src/decoding` and `src`
  - exec target repointed to `experiments/rescoring/generate_whisper_nbest.py`
- `run_rescore_whisper_nbest.sh`
  - repo-root detection changed to the release repo root
  - default N-best discovery now searches `experiments/rescoring/nbest/`
  - default output now lands in `experiments/rescoring/rescored/`
  - summary JSON output is written to `experiments/rescoring/summaries/`
  - `PYTHONPATH` now includes `src/decoding` and `src`
  - exec target repointed to `experiments/rescoring/rescore_whisper_nbest.py`
- `run_build_rescoring_report.sh`
  - default summaries now read from `experiments/rescoring/summaries/`
  - default output directory is `experiments/rescoring/`
  - exec target repointed to `experiments/rescoring/build_rescoring_report.py`
- `rescore_whisper_nbest.py`
  - file lookups now resolve from the repository root via `Path(__file__).resolve().parents[2]`
  - added `--summary-dir` for the report JSON destination
- `build_rescoring_report.py`
  - default summary locations and output directory now point to `experiments/rescoring/`

## Git ignore changes

The release repo `.gitignore` now excludes:

- `experiments/rescoring/nbest/`
- `experiments/rescoring/rescored/`

This keeps the generated Whisper N-best JSONL files and per-run rescoring outputs out of git while leaving the summary JSONs and markdown reports tracked.

## Remaining TODOs

- Choose the final public license if it is not already final.
- If you want the raw Whisper N-best JSONL files in an artifact store, regenerate them with `run_whisper_nbest.sh` and copy them outside git.
- The existing release intentionally keeps the large N-best JSONL files untracked; this is expected.
- Static import resolution against the released `src/` tree passed for the rescoring scripts.
