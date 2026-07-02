#!/usr/bin/env python3
"""Stratified speaker-balanced sample from a training JSONL.

Draws N_PER_SPEAKER utterances from each speaker (random, seed-fixed),
producing a reproducible manifest with speaker and sentence variability.
Gender is inferred from speaker_id (_F_ or _F suffix → F, _M_ or _M → M).
"""
import argparse
import json
import random
from pathlib import Path


def infer_gender(speaker_id: str) -> str:
    parts = speaker_id.split("_")
    for p in parts:
        if p == "F":
            return "F"
        if p == "M":
            return "M"
    return "U"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Source JSONL")
    parser.add_argument("--output",  required=True, help="Output JSONL")
    parser.add_argument("--n",       type=int, default=1000, help="Total samples")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    data = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]

    # Group by speaker
    by_speaker: dict[str, list] = {}
    for rec in data:
        by_speaker.setdefault(rec["speaker_id"], []).append(rec)

    n_speakers = len(by_speaker)
    base = args.n // n_speakers
    remainder = args.n % n_speakers

    # Sort speakers for determinism, assign extra slots to first `remainder` speakers
    speakers_sorted = sorted(by_speaker.keys())
    quota = {s: base + (1 if i < remainder else 0) for i, s in enumerate(speakers_sorted)}

    sampled = []
    for spk in speakers_sorted:
        pool = list(by_speaker[spk])
        rng.shuffle(pool)
        chosen = pool[: quota[spk]]
        gender = infer_gender(spk)
        for rec in chosen:
            sampled.append({
                "path":       rec["path"],
                "sentence":   rec["sentence"],
                "speaker_id": rec["speaker_id"],
                "lang":       rec["lang"],
                "gender":     gender,
                "duration":   rec["duration"],
            })

    # Final shuffle for mixing speakers (same seed, extended)
    rng.shuffle(sampled)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in sampled:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sampled)} utterances → {args.output}")
    print(f"  Speakers: {n_speakers}  |  Per speaker: {base} (+ {remainder} get one extra)")
    print(f"  Seed: {args.seed}")
    by_spk_out: dict[str, int] = {}
    for rec in sampled:
        by_spk_out[rec["speaker_id"]] = by_spk_out.get(rec["speaker_id"], 0) + 1
    for s in speakers_sorted:
        print(f"    {s}: {by_spk_out[s]}")


if __name__ == "__main__":
    main()
