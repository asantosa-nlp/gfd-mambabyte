#!/usr/bin/env python3
"""
Create a stratified 300-utterance search sample from the 1,000-utterance development
set for JV and SU.

Sampling strategy (in priority order):
  1. Cover all unique sentences at least once (228 unique for both JV and SU).
     For each unique sentence, the speaker is chosen to maximise speaker balance
     (greedy: always pick the speaker with fewest selections so far).
  2. Fill remaining slots (300 - 228 = 72) from unselected recordings,
     again greedy-balanced across speakers.
  3. Final list is shuffled with a fixed seed for reproducibility.

Output:
  data/speech_corpus/local_{jv,su}/gfd_search_300_stratified_seed42.jsonl

The complementary 700-utterance held-out set (for logit ratio analysis) is written to:
  data/speech_corpus/local_{jv,su}/gfd_logit_700_seed42.jsonl

Usage:
    python scripts/make_search_sample.py [--dry-run] [--seed 42] [--n 300]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_sample(records: list[dict], n: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Return (selected, held_out) from records using the coverage-then-balance strategy.

    Step 1 — unique sentence coverage:
        Group by normalised sentence text. For each unique sentence, pick one
        recording, cycling through speakers greedily (always pick from the speaker
        with fewest picks so far).

    Step 2 — fill to n:
        From the remaining (unselected) recordings, add greedily-balanced speakers
        until n total are selected.

    Step 3 — shuffle selected with fixed seed.
    """
    rng = random.Random(seed)

    # Group by (normalised sentence, speaker)
    by_sentence: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r.get("sentence", r.get("text", "")).strip().lower()
        by_sentence[key].append(r)

    # Sort unique sentences for determinism
    unique_sents = sorted(by_sentence.keys())
    n_unique = len(unique_sents)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    speaker_counts: dict[str, int] = defaultdict(int)

    # ── Step 1: Cover every unique sentence ───────────────────────────────────
    for sent in unique_sents:
        candidates = by_sentence[sent]
        # Sort candidates by (current speaker count asc, speaker_id for tie-break)
        candidates_sorted = sorted(
            candidates,
            key=lambda r: (speaker_counts[r.get("speaker_id", "?")],
                           r.get("speaker_id", "?"))
        )
        chosen = candidates_sorted[0]
        path_key = chosen["path"]
        selected.append(chosen)
        selected_ids.add(path_key)
        speaker_counts[chosen.get("speaker_id", "?")] += 1

    # ── Step 2: Fill remaining slots (n - n_unique) ───────────────────────────
    remaining = [r for r in records if r["path"] not in selected_ids]
    # Shuffle remaining for randomness within speaker groups
    rng.shuffle(remaining)

    needed = n - len(selected)
    while needed > 0 and remaining:
        # Find speaker with fewest selections
        min_count = min(speaker_counts[r.get("speaker_id", "?")] for r in remaining)
        # Among those with min count, pick one randomly (deterministic via rng)
        min_candidates = [r for r in remaining
                          if speaker_counts[r.get("speaker_id", "?")] == min_count]
        chosen = rng.choice(min_candidates)
        selected.append(chosen)
        selected_ids.add(chosen["path"])
        speaker_counts[chosen.get("speaker_id", "?")] += 1
        remaining = [r for r in remaining if r["path"] not in selected_ids]
        needed -= 1

    # ── Step 3: Shuffle selected with fixed seed ──────────────────────────────
    rng.shuffle(selected)

    held_out = [r for r in records if r["path"] not in selected_ids]

    return selected, held_out


def analyse(label: str, records: list[dict]) -> None:
    from collections import Counter
    speakers = Counter(r.get("speaker_id", "?") for r in records)
    sentences = [r.get("sentence", r.get("text", "")).strip().lower() for r in records]
    n_unique = len(set(sentences))
    print(f"  {label}: {len(records)} utts | {len(speakers)} speakers | {n_unique} unique sentences")
    counts = sorted(speakers.items())
    print(f"    Speaker dist: {dict(counts)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=300, help="target sample size")
    ap.add_argument("--dry-run", action="store_true", help="print stats but don't write files")
    args = ap.parse_args()

    for lang in ["jv", "su"]:
        src = Path(f"data/speech_corpus/local_{lang}/train_1000_seed42.jsonl")
        out_search = Path(f"data/speech_corpus/local_{lang}/gfd_search_300_stratified_seed42.jsonl")
        out_logit  = Path(f"data/speech_corpus/local_{lang}/gfd_logit_700_seed42.jsonl")

        records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

        print(f"\n{'='*60}")
        print(f"{lang.upper()} — source: {src.name} ({len(records)} utts)")
        analyse("Source 1k", records)

        selected, held_out = stratified_sample(records, n=args.n, seed=args.seed)

        print(f"\n  After stratified sampling (n={args.n}, seed={args.seed}):")
        analyse(f"Selected {len(selected)}", selected)
        analyse(f"Held-out {len(held_out)}", held_out)

        # Verify: all unique sentences covered
        src_sents = set(r.get("sentence", "").strip().lower() for r in records)
        sel_sents = set(r.get("sentence", "").strip().lower() for r in selected)
        uncovered = src_sents - sel_sents
        print(f"\n  Unique sentence coverage: {len(sel_sents)}/{len(src_sents)}")
        if uncovered:
            print(f"  WARNING: {len(uncovered)} sentences NOT covered: {list(uncovered)[:3]}...")
        else:
            print(f"  ✓ All {len(src_sents)} unique sentences covered in the {len(selected)}-utt sample")

        if not args.dry_run:
            out_search.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + "\n"
            )
            out_logit.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in held_out) + "\n"
            )
            print(f"\n  Written: {out_search}")
            print(f"  Written: {out_logit}")
        else:
            print(f"\n  [dry-run] would write: {out_search}")
            print(f"  [dry-run] would write: {out_logit}")

    print("\nDone.")


if __name__ == "__main__":
    main()
