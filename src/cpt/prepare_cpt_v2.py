#!/usr/bin/env python3
"""
scripts/prepare_cpt_v2.py

Extract all speech transcript sentences from OpenSLR and local_news datasets
for Javanese and Sundanese, then combine with existing cc100 + wikipedia text
corpora to produce augmented training files for MambaByte CPT v2.

Output files (in data/text_corpus/):
  transcripts_jv.txt   — raw sentences from OpenSLR JV + local_news JV
  transcripts_su.txt   — raw sentences from OpenSLR SU + local_news SU
  combined_jv_v2.txt   — cc100_jv + wikipedia_jv + transcripts_jv
  combined_su_v2.txt   — cc100_su + wikipedia_su + transcripts_su

Usage:
  python scripts/prepare_cpt_v2.py [--base-dir BASE_DIR]
"""

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_sentences(paths: list[str]) -> list[str]:
    """Extract and lightly clean sentences from JSONL manifest files."""
    sentences = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  [skip] not found: {path}")
            continue
        rows = load_jsonl(path)
        for r in rows:
            s = r.get("sentence", "").strip()
            if not s:
                continue
            # Skip lines that are predominantly ASCII/English
            # (same heuristic as the manifest filtering)
            non_ascii = sum(1 for c in s if ord(c) > 127)
            ascii_words = re.findall(r'\b[a-zA-Z]{3,}\b', s)
            eng_stopwords = {'the','is','are','was','were','and','that','this',
                             'for','with','from','have','has','been','not','but',
                             'they','their','what','which','when','where','who'}
            eng_hits = sum(1 for w in ascii_words if w.lower() in eng_stopwords)
            if eng_hits >= 2:
                continue  # skip English-contaminated sentences
            sentences.append(s)
    return sentences


def byte_count(path: str) -> int:
    return os.path.getsize(path)


def concat_files(inputs: list[str], output: str) -> int:
    """Concatenate text files, separating each with a newline. Returns total bytes."""
    total = 0
    with open(output, "wb") as out_f:
        for path in inputs:
            if not os.path.exists(path):
                print(f"  [skip] not found: {path}")
                continue
            with open(path, "rb") as in_f:
                data = in_f.read()
            out_f.write(data)
            # Ensure file ends with newline before next source
            if data and data[-1] != ord('\n'):
                out_f.write(b'\n')
            total += len(data)
            print(f"  + {path} ({len(data)/1e6:.1f} MB)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".",
                        help="Project base directory (default: current dir)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    corpus_dir = base / "data" / "text_corpus"
    speech_dir = base / "data" / "speech_corpus"

    # ── Source manifest files ─────────────────────────────────────────────────
    manifests = {
        "jv": [
            str(speech_dir / "openslr_jv" / "train.jsonl"),
            str(speech_dir / "openslr_jv" / "dev.jsonl"),
            str(speech_dir / "openslr_jv" / "test.jsonl"),
            str(speech_dir / "local_jv"   / "train.jsonl"),
            str(speech_dir / "local_jv"   / "dev.jsonl"),
            str(speech_dir / "local_jv"   / "test.jsonl"),
        ],
        "su": [
            str(speech_dir / "openslr_su" / "train.jsonl"),
            str(speech_dir / "openslr_su" / "dev.jsonl"),
            str(speech_dir / "openslr_su" / "test.jsonl"),
            str(speech_dir / "local_su"   / "train.jsonl"),
            str(speech_dir / "local_su"   / "dev.jsonl"),
            str(speech_dir / "local_su"   / "test.jsonl"),
        ],
    }

    for lang in ("jv", "su"):
        print(f"\n{'='*60}")
        print(f" Extracting transcripts: {lang.upper()}")
        print(f"{'='*60}")

        sentences = extract_sentences(manifests[lang])
        print(f"  Total sentences extracted: {len(sentences):,}")

        # Write transcript file (one sentence per line)
        transcript_path = str(corpus_dir / f"transcripts_{lang}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences) + "\n")
        transcript_bytes = os.path.getsize(transcript_path)
        print(f"  Written: {transcript_path} ({transcript_bytes/1e6:.2f} MB)")

        # Print sample sentences
        print(f"\n  Sample extracted sentences:")
        import random; random.seed(42)
        for s in random.sample(sentences, min(5, len(sentences))):
            print(f"    {s[:80]}")

    print(f"\n{'='*60}")
    print(" Creating combined training files")
    print(f"{'='*60}")

    for lang in ("jv", "su"):
        sources = [
            str(corpus_dir / f"cc100_{lang}.txt"),
            str(corpus_dir / f"wikipedia_{lang}.txt"),
            str(corpus_dir / f"transcripts_{lang}.txt"),
        ]
        output = str(corpus_dir / f"combined_{lang}_v2.txt")
        print(f"\n[{lang.upper()}] → {output}")
        total = concat_files(sources, output)
        n_bytes = os.path.getsize(output)
        n_windows = max(0, n_bytes - 1024)  # approximate seq_len=1024 windows
        print(f"  Total size  : {n_bytes/1e6:.1f} MB  ({n_bytes:,} bytes)")
        print(f"  ~Windows    : {n_windows:,}  (seq_len=1024, before shuffle)")

    print(f"\n{'='*60}")
    print(" Summary")
    print(f"{'='*60}")
    for lang in ("jv", "su"):
        src_v1 = str(corpus_dir / f"cc100_{lang}.txt")
        src_v2 = str(corpus_dir / f"combined_{lang}_v2.txt")
        b1 = os.path.getsize(src_v1)
        b2 = os.path.getsize(src_v2)
        print(f"  {lang.upper()} v1 (cc100 only)  : {b1/1e6:6.1f} MB")
        print(f"  {lang.upper()} v2 (combined)     : {b2/1e6:6.1f} MB  (+{(b2-b1)/1e6:.1f} MB, +{(b2/b1-1)*100:.0f}%)")

    print("\nDone. Next: create configs/cpt_jv_v2.yaml and configs/cpt_su_v2.yaml")


if __name__ == "__main__":
    main()
