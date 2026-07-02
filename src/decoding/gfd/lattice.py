"""gfd/lattice.py

Convert ASR BPE token logits into byte-level transitions.
Adapted from blt-gfd-jvsu-asr/src/fusion/lattice.py.
"""
from __future__ import annotations

from typing import Any

import torch


def _get_token_bytes(tokenizer, token_id: int) -> bytes:
    try:
        s = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return b""
    if s is None:
        return b""
    return s.encode("utf-8")


def byte_automaton_expand(
    next_token_logits: torch.Tensor,
    tokenizer,
    prev_byte_prefix: bytes,
    top_k: int = 32,
) -> list[dict[str, Any]]:
    """Expand top-k ASR token candidates into byte-level transitions.

    Each dict: next_token, next_bytes, asr_logp, is_eos.
    """
    log_probs = torch.log_softmax(next_token_logits, dim=-1)
    top = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]))
    out: list[dict[str, Any]] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    for lp, tid in zip(top.values.tolist(), top.indices.tolist()):
        if tid == eos:
            out.append({"next_token": tid, "next_bytes": b"", "asr_logp": lp, "is_eos": True})
            continue
        bs = _get_token_bytes(tokenizer, tid)
        out.append({"next_token": tid, "next_bytes": bs, "asr_logp": lp, "is_eos": False})
    return out
