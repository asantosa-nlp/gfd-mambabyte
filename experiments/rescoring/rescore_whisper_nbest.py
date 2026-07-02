#!/usr/bin/env python3
"""
Rescore fixed Whisper N-best lists with the adapted MambaByte LM.

This is the post-hoc baseline requested for reviewer comment R2Q2b:
the same adapted LM and the same fusion coefficients as GFD are applied
after first-pass Whisper decoding, instead of during search.

Outputs:
  - per-utterance rescored JSONL
  - summary JSON
  - comparison markdown helpers are produced by the companion driver
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_text(text: str, form: str = "NFC") -> str:
    text = unicodedata.normalize(form, text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_progress(rescored_path: Path, state_path: Path) -> tuple[int, int]:
    """Return (next_index, truncate_to_bytes) for resuming a JSONL run.

    If a state file exists and looks valid, prefer it. Otherwise, scan the JSONL
    output and count parseable rows. If the last line is partial/corrupted, the
    returned truncate offset lets the caller trim the file before appending.
    """
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            next_index = int(state.get("next_index", 0))
            if next_index >= 0:
                return next_index, int(state.get("truncate_to_bytes", rescored_path.stat().st_size if rescored_path.exists() else 0))
        except Exception:
            pass

    if not rescored_path.exists():
        return 0, 0

    next_index = 0
    truncate_to = 0
    with open(rescored_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                truncate_to = offset
                break
            try:
                json.loads(line.decode("utf-8").strip())
            except Exception:
                truncate_to = offset
                break
            next_index += 1
            truncate_to = f.tell()
    return next_index, truncate_to


def save_state(
    state_path: Path,
    *,
    next_index: int,
    truncate_to_bytes: int,
    utt_id: str | None = None,
    status: str = "running",
    error: str | None = None,
) -> None:
    payload = {
        "status": status,
        "next_index": next_index,
        "truncate_to_bytes": truncate_to_bytes,
        "utt_id": utt_id,
        "error": error,
        "timestamp": time.time(),
    }
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)


def has_repeat_suffix(seq: bytes, min_unit: int, max_unit: int, min_reps: int) -> bool:
    try:
        seq_lower = seq.decode("utf-8", errors="replace").lower().encode("utf-8")
    except Exception:
        seq_lower = seq
    lower_safe = len(seq_lower) == len(seq)
    for w in range(min_unit, max_unit + 1):
        needed = w * min_reps
        if len(seq) < needed:
            continue
        unit = seq[-w:]
        if seq[-needed:] == unit * min_reps:
            return True
        if lower_safe:
            unit_l = seq_lower[-w:]
            if seq_lower[-needed:] == unit_l * min_reps:
                return True
    return False


def has_long_word(seq: bytes, max_word_bytes: int) -> bool:
    last_space = seq.rfind(b" ")
    word_len = len(seq) if last_space == -1 else len(seq) - last_space - 1
    return word_len > max_word_bytes


def load_cfg(lang: str) -> dict:
    if lang == "jv":
        path = REPO_ROOT / "configs/mb_jv_zs_large_v5cleanS20k_ckptsel1k.yaml"
    elif lang == "su":
        path = REPO_ROOT / "configs/mb_su_zs_large_v5cleanS20k_gs_lm80_ag20_minb15.yaml"
    else:
        raise ValueError(f"Unsupported lang: {lang}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_zero_shot_predictions(lang: str) -> list[dict]:
    if lang == "jv":
        path = REPO_ROOT / "experiments/rescoring/baseline_predictions/jv_zero_shot_predictions.json"
    else:
        path = REPO_ROOT / "experiments/rescoring/baseline_predictions/su_zero_shot_predictions.json"
    return load_json(path)


def load_gfd_predictions(lang: str) -> list[dict]:
    if lang == "jv":
        path = REPO_ROOT / "experiments/rescoring/baseline_predictions/jv_gfd_predictions.json"
    else:
        path = REPO_ROOT / "experiments/rescoring/baseline_predictions/su_gfd_predictions.json"
    return load_json(path)


def oracle_best_idx(reference: str, hypotheses: list[dict]) -> tuple[int, float, float]:
    best_i = 0
    best_wer = math.inf
    best_cer = math.inf
    for i, hyp in enumerate(hypotheses):
        hyp_n = normalize_text(hyp["text"])
        wer = float(jiwer.wer(reference, hyp_n)) if reference else 0.0
        cer = float(jiwer.cer(reference, hyp_n)) if reference else 0.0
        if wer < best_wer:
            best_i = i
            best_wer = wer
            best_cer = cer
    return best_i, best_wer, best_cer


def score_bytes_from_lm(lm, root, byte_seq: bytes, log_floor: float = -30.0) -> tuple[float, int]:
    total = 0.0
    forwards = 0
    node = root
    for b in byte_seq:
        lp = lm.byte_logprob_from_node(node, b)
        total += max(float(lp), log_floor)
        node, _ = lm.extend_prefix_cache(node, b)
        forwards += 1
    return total, forwards


def permutation_test_pvalue(deltas: list[float], seed: int = 42, num_perm: int = 10000) -> tuple[float, float]:
    """Two-sided matched-pairs sign-flip permutation test on mean delta."""
    rng = random.Random(seed)
    obs = sum(deltas) / max(len(deltas), 1)
    if not deltas:
        return 1.0, obs
    ge = 0
    abs_obs = abs(obs)
    for _ in range(num_perm):
        s = 0.0
        for d in deltas:
            s += d if rng.random() < 0.5 else -d
        if abs(s / len(deltas)) >= abs_obs:
            ge += 1
    p = (ge + 1) / (num_perm + 1)
    return p, obs


def main() -> int:
    ap = argparse.ArgumentParser(description="Rescore Whisper N-best with adapted MambaByte")
    ap.add_argument("--lang", required=True, choices=["jv", "su"])
    ap.add_argument("--nbest-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--summary-dir",
        default=None,
        help="Directory to write the summary JSON (default: --out-dir)",
    )
    ap.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-permutations", type=int, default=10000)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the latest completed row in the output JSONL (default: True).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N utterances (default: 1).",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.lang)
    norm_cfg = cfg["eval"]["text_normalization"]
    fusion_cfg = cfg["fusion"]
    lm_cfg = cfg["lm"]

    lam = float(fusion_cfg["lambda"])
    alpha = float(fusion_cfg["lm_logit_alpha"])
    min_bytes = int(fusion_cfg["min_output_bytes"])
    max_bytes = int(fusion_cfg["max_output_bytes"])
    rep = fusion_cfg["heuristic_filters"]["rep_detect"]
    long_word = fusion_cfg["heuristic_filters"]["long_word"]
    log_floor = float(fusion_cfg.get("log_prob_floor", -30.0))

    if args.lang == "jv":
        ref_dir = REPO_ROOT / "data/speech_corpus/local_jv/test.jsonl"
    else:
        ref_dir = REPO_ROOT / "data/speech_corpus/local_su/test.jsonl"
    refs = load_jsonl(ref_dir)
    nbest_rows = load_jsonl(args.nbest_file)
    if len(refs) != len(nbest_rows):
        raise RuntimeError(f"Reference/test rows ({len(refs)}) and N-best rows ({len(nbest_rows)}) do not match")

    # Align with the existing source material
    zero_preds = load_zero_shot_predictions(args.lang)
    gfd_preds = load_gfd_predictions(args.lang)
    if len(zero_preds) != len(refs) or len(gfd_preds) != len(refs):
        raise RuntimeError("Reference and baseline prediction lengths do not match")
    n = len(refs)

    os.makedirs(args.out_dir, exist_ok=True)
    rescored_path = Path(args.out_dir) / f"{args.lang}_rescored.jsonl"
    summary_dir = Path(args.summary_dir) if args.summary_dir else Path(args.out_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{args.lang}_summary.json"
    state_path = Path(args.out_dir) / f"{args.lang}_rescored.state.json"

    from gfd.lm_adapters import MambaByteAdapter

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = args.device
    lm = MambaByteAdapter(
        model_name_or_path=lm_cfg["model_name_or_path"],
        lora_ckpt=lm_cfg["lora_ckpt"],
        device=device,
        dtype=dtype_map[args.dtype],
    )

    out_rows: list[dict] = []
    total_rescore_wer = total_rescore_cer = 0.0
    total_oracle_wer = total_oracle_cer = 0.0
    total_zero_wer = total_zero_cer = 0.0
    total_gfd_wer = total_gfd_cer = 0.0
    deltas: list[float] = []

    start_index = 0
    truncate_to = 0
    if args.resume:
        start_index, truncate_to = load_progress(rescored_path, state_path)
        if start_index > 0:
            print(f"[rescore] Resuming from row {start_index + 1}/{n}")
        if rescored_path.exists() and truncate_to < rescored_path.stat().st_size:
            with open(rescored_path, "r+b") as f:
                f.truncate(truncate_to)

    if args.resume and rescored_path.exists() and start_index > 0:
        with open(rescored_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                out_rows.append(row)
                total_rescore_wer += row["selected"]["wer"]
                total_rescore_cer += row["selected"]["cer"]
                total_oracle_wer += row["oracle"]["wer"]
                total_oracle_cer += row["oracle"]["cer"]
                total_zero_wer += row["zero_shot"]["wer"]
                total_zero_cer += row["zero_shot"]["cer"]
                total_gfd_wer += row["gfd"]["wer"]
                total_gfd_cer += row["gfd"]["cer"]
                deltas.append(row["gfd"]["wer"] - row["selected"]["wer"])

    mode = "a" if args.resume and start_index > 0 else "w"
    try:
        with open(rescored_path, mode, encoding="utf-8") as f:
            for idx, (ref_row, nb_row, zero_row, gfd_row) in enumerate(zip(refs, nbest_rows, zero_preds, gfd_preds)):
                if idx < start_index:
                    continue

                utt_id = nb_row.get("utt_id", f"utt_{idx:04d}")
                if args.progress_every > 0 and (idx % args.progress_every == 0):
                    print(
                        f"[rescore] {idx + 1}/{n} {utt_id} "
                        f"(processing on {device}, λ={lam:.2f}, α={alpha:.2f})",
                        flush=True,
                    )

                reference_raw = ref_row["sentence"]
                reference = normalize_text(reference_raw, norm_cfg["unicode_form"])

                candidates = []
                root = lm.init_prefix_cache()
                for cand in nb_row["hypotheses"]:
                    raw_text = cand["text"]
                    raw_bytes = raw_text.encode("utf-8")
                    n_bytes = len(raw_bytes)
                    valid = True
                    invalid_reason = None
                    if n_bytes < min_bytes:
                        valid = False
                        invalid_reason = "too_short"
                    elif n_bytes > max_bytes:
                        valid = False
                        invalid_reason = "too_long"
                    elif has_repeat_suffix(raw_bytes, int(rep["min_unit"]), int(rep["max_unit"]), int(rep["min_reps"])):
                        valid = False
                        invalid_reason = "repeat_suffix"
                    elif has_long_word(raw_bytes, int(long_word["max_word_bytes"])):
                        valid = False
                        invalid_reason = "long_word"

                    if valid:
                        lm_logprob, lm_forwards = score_bytes_from_lm(lm, root, raw_bytes, log_floor=log_floor)
                        fused = (1.0 - lam) * float(cand["whisper_logprob"]) + lam * alpha * lm_logprob
                    else:
                        lm_logprob = None
                        lm_forwards = 0
                        fused = float("-inf")

                    candidates.append({
                        "rank": cand["rank"],
                        "text": raw_text,
                        "whisper_logprob": float(cand["whisper_logprob"]),
                        "byte_len": n_bytes,
                        "valid": valid,
                        "invalid_reason": invalid_reason,
                        "lm_logprob": lm_logprob,
                        "lm_forwards": lm_forwards,
                        "fused_score": fused,
                        "normalized": normalize_text(raw_text, norm_cfg["unicode_form"]),
                    })

                valid_idxs = [i for i, c in enumerate(candidates) if c["valid"]]
                if valid_idxs:
                    best_idx = max(valid_idxs, key=lambda i: candidates[i]["fused_score"])
                else:
                    best_idx = max(range(len(candidates)), key=lambda i: candidates[i]["whisper_logprob"])
                best = candidates[best_idx]

                oracle_idx, oracle_wer, oracle_cer = oracle_best_idx(reference, candidates)
                oracle_text = candidates[oracle_idx]["text"]
                best_wer = float(jiwer.wer(reference, best["normalized"])) if reference else 0.0
                best_cer = float(jiwer.cer(reference, best["normalized"])) if reference else 0.0

                zero_hyp = normalize_text(zero_row["hypothesis"], norm_cfg["unicode_form"])
                zero_wer = float(jiwer.wer(reference, zero_hyp)) if reference else 0.0
                zero_cer = float(jiwer.cer(reference, zero_hyp)) if reference else 0.0

                gfd_hyp = normalize_text(gfd_row["hypothesis"], norm_cfg["unicode_form"])
                gfd_wer = float(jiwer.wer(reference, gfd_hyp)) if reference else 0.0
                gfd_cer = float(jiwer.cer(reference, gfd_hyp)) if reference else 0.0

                total_rescore_wer += best_wer
                total_rescore_cer += best_cer
                total_oracle_wer += oracle_wer
                total_oracle_cer += oracle_cer
                total_zero_wer += zero_wer
                total_zero_cer += zero_cer
                total_gfd_wer += gfd_wer
                total_gfd_cer += gfd_cer
                deltas.append(gfd_wer - best_wer)

                row = {
                    "utt_id": utt_id,
                    "speaker_id": nb_row.get("speaker_id"),
                    "reference_raw": reference_raw,
                    "reference": reference,
                    "selected": {
                        "rank": best["rank"],
                        "text": best["text"],
                        "normalized": best["normalized"],
                        "whisper_logprob": best["whisper_logprob"],
                        "lm_logprob": best["lm_logprob"],
                        "fused_score": best["fused_score"],
                        "valid": best["valid"],
                        "invalid_reason": best["invalid_reason"],
                        "wer": best_wer,
                        "cer": best_cer,
                    },
                    "oracle": {
                        "rank": candidates[oracle_idx]["rank"],
                        "text": oracle_text,
                        "normalized": candidates[oracle_idx]["normalized"],
                        "wer": oracle_wer,
                        "cer": oracle_cer,
                    },
                    "zero_shot": {
                        "hypothesis": zero_row["hypothesis"],
                        "wer": zero_wer,
                        "cer": zero_cer,
                    },
                    "gfd": {
                        "hypothesis": gfd_row["hypothesis"],
                        "wer": gfd_wer,
                        "cer": gfd_cer,
                    },
                    "candidates": candidates,
                }
                out_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
                if (idx + 1) % max(args.progress_every, 1) == 0:
                    print(
                        f"[rescore] {idx + 1}/{n} {utt_id} done "
                        f"(selected rank {best['rank']}, WER={best_wer*100:.2f}%)",
                        flush=True,
                    )
                save_state(
                    state_path,
                    next_index=idx + 1,
                    truncate_to_bytes=f.tell(),
                    utt_id=utt_id,
                    status="running",
                )
    except Exception as e:
        # State has already been updated per row; re-raise after making sure
        # the last known progress persists.
        save_state(
            state_path,
            next_index=len(out_rows),
            truncate_to_bytes=rescored_path.stat().st_size if rescored_path.exists() else 0,
            status="error",
            error=str(e),
        )
        raise

    n = len(out_rows)
    mean_rescore_wer = total_rescore_wer / n
    mean_rescore_cer = total_rescore_cer / n
    mean_oracle_wer = total_oracle_wer / n
    mean_oracle_cer = total_oracle_cer / n
    mean_zero_wer = total_zero_wer / n
    mean_zero_cer = total_zero_cer / n
    mean_gfd_wer = total_gfd_wer / n
    mean_gfd_cer = total_gfd_cer / n

    p_value, mean_delta = permutation_test_pvalue(deltas, seed=args.seed, num_perm=args.num_permutations)
    summary = {
        "lang": args.lang,
        "n_utterances": n,
        "lambda": lam,
        "alpha": alpha,
        "min_output_bytes": min_bytes,
        "max_output_bytes": max_bytes,
        "zero_shot": {"wer": mean_zero_wer, "cer": mean_zero_cer},
        "rescoring": {"wer": mean_rescore_wer, "cer": mean_rescore_cer},
        "oracle": {"wer": mean_oracle_wer, "cer": mean_oracle_cer},
        "gfd": {"wer": mean_gfd_wer, "cer": mean_gfd_cer},
        "paired_delta_gfd_minus_rescoring": mean_delta,
        "permutation_test": {
            "seed": args.seed,
            "num_permutations": args.num_permutations,
            "p_value": p_value,
        },
        "nbest_file": args.nbest_file,
    }
    save_json(summary_path, summary)
    save_state(
        state_path,
        next_index=n,
        truncate_to_bytes=rescored_path.stat().st_size if rescored_path.exists() else 0,
        status="done",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
