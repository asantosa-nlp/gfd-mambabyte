"""gfd/entropy_gate.py

Entropy-gated LM invocation rule (Eq. 4 of the manuscript).
Adapted from blt-gfd-jvsu-asr/src/fusion/entropy_gate.py.
"""
from __future__ import annotations

import math
from typing import Any


class EntropyGate:
    """Decides whether to invoke the byte-level LM at a given byte position.

    Modes:
      * 'always'    — always invoke (τ_G = 0)
      * 'never'     — never invoke (no fusion)
      * 'threshold' — invoke iff H_t > τ_G (nats)
    """

    def __init__(self, cfg: dict, lm: Any | None):
        self.enabled: bool = cfg.get("enabled", False)
        self.mode: str = cfg.get("mode", "always")
        self.tau_G: float = float(cfg.get("tau_G", 0.0))
        self.lm = lm
        self._invocations = 0
        self._skips = 0

    def should_invoke(self, byte_prefix: bytes) -> bool:
        if not self.enabled or self.mode == "always":
            return self._record(True)
        if self.mode == "never":
            return self._record(False)
        if self.mode == "threshold":
            h = self._patch_entropy(byte_prefix)
            return self.should_invoke_entropy(h)
        raise ValueError(f"unknown gate mode: {self.mode}")

    def should_invoke_entropy(self, entropy: float) -> bool:
        if not self.enabled or self.mode == "always":
            return self._record(True)
        if self.mode == "never":
            return self._record(False)
        if self.mode == "threshold":
            return self._record(float(entropy) > self.tau_G)
        raise ValueError(f"unknown gate mode: {self.mode}")

    def _patch_entropy(self, byte_prefix: bytes) -> float:
        if self.lm is None or not hasattr(self.lm, "next_byte_entropy"):
            return math.inf
        return float(self.lm.next_byte_entropy(byte_prefix))

    def stats(self) -> dict[str, int]:
        return {
            "invocations": self._invocations,
            "skips": self._skips,
            "total": self._invocations + self._skips,
        }

    def _record(self, invoke: bool) -> bool:
        if invoke:
            self._invocations += 1
        else:
            self._skips += 1
        return invoke
