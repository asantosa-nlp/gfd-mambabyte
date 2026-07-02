The scoring component is the evaluation runner. The main entry point is `run_exp_mb.py`, which loads Whisper + MambaByte, runs GFD on the blind/search split, normalizes text, and computes WER/CER.
