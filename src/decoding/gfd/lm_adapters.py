"""gfd/lm_adapters.py

MambaByte adapter for Generative Fusion Decoding.

Key architectural advantage over BLT:
  MambaByte is a pure SSM (state-space model). Every forward step carries
  a fixed-size recurrent state h_t of shape [n_layers, d_state]. Extending
  a prefix by one byte costs O(1) — one SSM step — regardless of sequence
  length. There is no need for the "lazy prefetch" trick used in the BLT
  adapter; every extension is already a single cheap step.

Prefix trie structure:
  Each trie node stores the SSM state (conv_states + ssm_states from
  mamba_ssm InferenceParams) and the resulting next-byte log_probs.
  Child creation = restore state → run one step → store new state + logits.

Usage:
    adapter = MambaByteAdapter(
        model_name_or_path="JunxiongWang/MambaByte_PG19_972M",
        lora_ckpt=None,    # or path to LoRA checkpoint from CPT
        device="cuda",
        dtype=torch.bfloat16,
    )
    root = adapter.init_prefix_cache()
    child, hit = adapter.extend_prefix_cache(root, byte_val=65)  # 'A'
    logp = adapter.byte_logprob_from_node(root, 65)
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


def _patch_triton_layer_norm_for_sm70():
    """Replace mamba_ssm's Triton RMSNorm kernel with a PyTorch-native fallback.

    V100 (sm_70) does not support bfloat16 PTX instructions, so the Triton
    layer_norm kernel crashes at JIT compile time. This monkey-patch provides
    a numerically equivalent PyTorch eager implementation and must be applied
    to every mamba_ssm module that holds its own binding to layer_norm_fn.
    """
    try:
        def _pytorch_layer_norm_fn(
            x, weight, bias=None, residual=None, x1=None, weight1=None,
            bias1=None, eps=1e-6, dropout_p=0.0, rowscale=None,
            prenorm=False, residual_in_fp32=False, is_rms_norm=False,
            num_groups=1, norm_before_gate=True, **kwargs,
        ):
            orig_dtype = x.dtype
            if residual is not None:
                residual = (residual.float() + x.float()) if residual_in_fp32 else (residual + x)
                x = residual.to(orig_dtype)
            if is_rms_norm:
                xf = x.float()
                rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                out = (xf * rms).to(orig_dtype) * weight
                if bias is not None:
                    out = out + bias
            else:
                out = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
            if prenorm:
                return out, residual if residual is not None else x
            return out

        def _pytorch_rms_norm_fn(*args, **kwargs):
            kwargs["is_rms_norm"] = True
            return _pytorch_layer_norm_fn(*args, **kwargs)

        import mamba_ssm.ops.triton.layer_norm as _ln_mod
        import mamba_ssm.modules.block as _block_mod
        import mamba_ssm.models.mixer_seq_simple as _mixer_mod
        import mamba_ssm.modules.mamba_simple as _mamba_simple_mod

        for _mod in (_ln_mod, _block_mod, _mixer_mod, _mamba_simple_mod):
            if hasattr(_mod, "layer_norm_fn"):
                _mod.layer_norm_fn = _pytorch_layer_norm_fn
            if hasattr(_mod, "rms_norm_fn"):
                _mod.rms_norm_fn = _pytorch_rms_norm_fn
    except Exception:
        pass  # If mamba_ssm isn't installed, nothing to patch


# ---------------------------------------------------------------------------
# Prefix trie node
# ---------------------------------------------------------------------------

@dataclass
class MambaBytePrefixNode:
    """One node in the MambaByte prefix trie.

    states is a dict mapping layer_idx -> (conv_state_cpu, ssm_state_cpu).
    Shapes (batch dim removed for storage):
      conv_state: [d_model, d_conv-1]  e.g. [3584, 4]
      ssm_state:  [d_model, d_state]   e.g. [3584, 16]
    Stored on CPU; moved to GPU only during extend calls.

    Verified against mamba-ssm 2.2.4 / MambaByte_PG19_972M:
      key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
      vocab_size = 256, byte_offset = 0, 48 layers.
    """
    byte_seq: bytes
    # Per-layer SSM states: {layer_idx: (conv_state_cpu, ssm_state_cpu)}
    states: dict
    # Next-byte distribution (cpu tensor, shape [vocab_size=256])
    log_probs: torch.Tensor
    entropy: float = 0.0
    children: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MambaByteAdapter
# ---------------------------------------------------------------------------

class MambaByteAdapter:
    """Wraps MambaByte for byte-level GFD prefix scoring.

    Unlike the BLT adapter which re-processes the full prefix on each cache
    miss, this adapter carries the SSM state forward one byte at a time.
    Every extend_prefix_cache call is a single autoregressive step O(1).
    """

    _log_floor: float = -30.0

    def __init__(
        self,
        model_name_or_path: str = "JunxiongWang/MambaByte_PG19_972M",
        lora_ckpt: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.model_name_or_path = model_name_or_path

        print(f"[lm_adapters] Loading MambaByte from {model_name_or_path} ...")
        self.model = self._load_model(model_name_or_path, device, dtype)

        if lora_ckpt is not None:
            print(f"[lm_adapters] Loading LoRA weights from {lora_ckpt} ...")
            self._load_lora(lora_ckpt)

        self.model.eval()

        # Vocabulary: MambaByte uses raw bytes. Token ids 0..255 map to bytes
        # 0..255. Offset may differ depending on model config — verify on init.
        self._byte_token_offset = self._detect_byte_offset()
        self.vocab_size = self.model.config.vocab_size
        print(f"[lm_adapters] MambaByteAdapter ready. "
              f"vocab={self.vocab_size}  byte_offset={self._byte_token_offset}")

    # ------------------------------------------------------------------
    # Public API (same interface as BLTLoRAAdapter)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def init_prefix_cache(self) -> MambaBytePrefixNode:
        """Create root node by running the model on an empty prefix."""
        return self._forward_step_full(b"")

    @torch.inference_mode()
    def extend_prefix_cache(
        self, prefix_node: MambaBytePrefixNode, next_byte: int
    ) -> tuple[MambaBytePrefixNode, bool]:
        """Extend prefix by one byte using O(1) SSM step.

        Returns (child_node, cache_hit). If child already exists: cache hit.
        If not: run one SSM step from parent state, create and cache child.
        """
        key = int(next_byte)
        cached = prefix_node.children.get(key)
        if cached is not None:
            return cached, True
        child = self._step_from_node(prefix_node, next_byte)
        prefix_node.children[key] = child
        return child, False

    def byte_logprob_from_node(
        self, prefix_node: MambaBytePrefixNode, next_byte: int
    ) -> float:
        idx = int(next_byte) + self._byte_token_offset
        return float(prefix_node.log_probs[idx].item())

    def entropy_from_node(self, prefix_node: MambaBytePrefixNode) -> float:
        return float(prefix_node.entropy)

    @torch.inference_mode()
    def batch_extend_from_node(
        self, prefix_node: MambaBytePrefixNode, next_bytes: list[int]
    ) -> None:
        """Create children for multiple next_bytes from the same parent.

        For Mamba, each step is independent from the same parent state,
        so we process each byte separately (SSM step cannot be trivially
        batched over different continuations — each shares the same prefix
        state but diverges after one step).
        """
        for b in next_bytes:
            if int(b) not in prefix_node.children:
                child = self._step_from_node(prefix_node, b)
                prefix_node.children[int(b)] = child

    @torch.inference_mode()
    def ensure_child_node(
        self, parent_node: MambaBytePrefixNode, byte_val: int
    ) -> None:
        """Ensure child for byte_val exists (creates if missing)."""
        if int(byte_val) not in parent_node.children:
            child = self._step_from_node(parent_node, byte_val)
            parent_node.children[int(byte_val)] = child

    def create_approx_child(
        self, parent_node: MambaBytePrefixNode, byte_val: int
    ) -> None:
        """Create an approximate child without a forward pass.
        Uses parent's argmax as a peaked distribution for the child.
        """
        if int(byte_val) in parent_node.children:
            return
        top_idx = int(parent_node.log_probs.argmax().item())
        approx_log_probs = torch.full_like(parent_node.log_probs, self._log_floor)
        approx_log_probs[top_idx] = 0.0
        # SSM state is copied from parent (approximate — no actual step)
        parent_node.children[int(byte_val)] = MambaBytePrefixNode(
            byte_seq=parent_node.byte_seq + bytes([byte_val]),
            states=copy.deepcopy(parent_node.states),
            log_probs=approx_log_probs,
            entropy=0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _step_from_node(
        self, node: MambaBytePrefixNode, next_byte: int
    ) -> MambaBytePrefixNode:
        """Run one autoregressive SSM step from node's state.

        1. Restore InferenceParams from node's stored state tensors.
        2. Run model on [next_byte] with seqlen_offset = len(node.byte_seq).
        3. Extract new state and logits.
        4. Return new MambaBytePrefixNode.
        """
        from mamba_ssm.utils.generation import InferenceParams

        params = self._restore_inference_params(node)
        token_id = int(next_byte) + self._byte_token_offset
        input_ids = torch.tensor(
            [[token_id]], device=self.device, dtype=torch.long
        )
        out = self.model(
            input_ids=input_ids,
            inference_params=params,
        )
        raw = out.logits if hasattr(out, 'logits') else out[0]
        logits = raw[0, -1]  # [vocab_size]
        safe = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        log_probs = torch.log_softmax(safe.float(), dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum().item())

        return MambaBytePrefixNode(
            byte_seq=node.byte_seq + bytes([next_byte]),
            states=self._extract_states(params),
            log_probs=log_probs.cpu(),
            entropy=entropy,
        )

    @torch.inference_mode()
    def _forward_step_full(self, prefix: bytes) -> MambaBytePrefixNode:
        """Process an entire byte sequence from scratch to get the root node.
        Called once per utterance to initialize the trie root.
        For an empty prefix: single forward pass of a special start token or
        an empty sequence (model-dependent — adjust if needed).
        """
        from mamba_ssm.utils.generation import InferenceParams

        if len(prefix) == 0:
            # Run on a single BOS-equivalent token to seed the SSM state.
            # MambaByte_PG19 was trained without explicit BOS; we use token 0.
            # TODO: verify with model card if a different seed token is used.
            token_ids = torch.zeros(1, 1, device=self.device, dtype=torch.long)
        else:
            ids = [b + self._byte_token_offset for b in prefix]
            token_ids = torch.tensor([ids], device=self.device, dtype=torch.long)

        params = InferenceParams(
            max_seqlen=len(prefix) + 512,
            max_batch_size=1,
        )
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            out = self.model(
                input_ids=token_ids,
                inference_params=params,
            )
        raw = out.logits if hasattr(out, 'logits') else out[0]
        logits = raw[0, -1]
        safe = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        log_probs = torch.log_softmax(safe.float(), dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum().item())

        return MambaBytePrefixNode(
            byte_seq=prefix,
            states=self._extract_states(params),
            log_probs=log_probs.cpu(),
            entropy=entropy,
        )

    def _restore_inference_params(self, node: MambaBytePrefixNode):
        """Reconstruct InferenceParams from stored cpu state tuples.

        mamba-ssm 2.2.4 API (verified):
          key_value_memory_dict[layer_idx] = (conv_state, ssm_state)
          conv_state: [batch, d_model, d_conv-1]
          ssm_state:  [batch, d_model, d_state]
        """
        from mamba_ssm.utils.generation import InferenceParams

        seqlen_offset = len(node.byte_seq)
        params = InferenceParams(
            max_seqlen=seqlen_offset + 512,
            max_batch_size=1,
        )
        params.seqlen_offset = seqlen_offset
        for layer_idx, (conv_state, ssm_state) in node.states.items():
            params.key_value_memory_dict[layer_idx] = (
                conv_state.unsqueeze(0).to(self.device),
                ssm_state.unsqueeze(0).to(self.device),
            )
        return params

    def _extract_states(self, params) -> dict:
        """Extract per-layer state tuples from InferenceParams to cpu.

        Returns {layer_idx: (conv_state_cpu, ssm_state_cpu)} with batch
        dim squeezed out for compact CPU storage.
        """
        states: dict = {}
        for layer_idx, (conv_state, ssm_state) in params.key_value_memory_dict.items():
            states[layer_idx] = (conv_state.squeeze(0).cpu(), ssm_state.squeeze(0).cpu())
        return states

    def _detect_byte_offset(self) -> int:
        """Detect the token ID offset for raw bytes.
        MambaByte_PG19 uses token IDs 0..255 directly (offset=0).
        Subclasses/checkpoints may differ.
        """
        # TODO: verify by checking model.config or embedding table size
        return 0

    def _load_model(self, name_or_path: str, device: str, dtype: torch.dtype):
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        # On sm_70 (V100) bfloat16 Triton kernels fail PTX compilation.
        # Patch both layer_norm and selective-scan fast path before loading.
        if device != "cpu":
            major, _ = torch.cuda.get_device_capability()
            if major < 8:
                _patch_triton_layer_norm_for_sm70()
        model = MambaLMHeadModel.from_pretrained(
            name_or_path,
            device=device,
            dtype=dtype,
        )
        for module in model.modules():
            if hasattr(module, "use_fast_path"):
                module.use_fast_path = False
        return model

    def _load_lora(self, lora_ckpt: str) -> None:
        """Load LoRA adapter from checkpoint (directory or .pt file from cpt/train.py)."""
        import os
        if os.path.isdir(lora_ckpt):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_ckpt)
            self.model = self.model.merge_and_unload()
        else:
            # .pt saved by cpt/train.py: inject LoRA layers first, then load weights.
            # GFD inference always passes inference_params, which triggers the mamba_ssm
            # slow path — so nn.Linear.forward() is called and LoRA adapters are active.
            from peft import LoraConfig, inject_adapter_in_model
            ckpt = torch.load(lora_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("lora_state_dict", ckpt)
            lora_cfg = ckpt.get("lora_config", {})
            rank = lora_cfg.get("rank", 16)
            alpha = float(lora_cfg.get("alpha", 32.0))
            dropout = float(lora_cfg.get("dropout", 0.05))
            target_modules = lora_cfg.get(
                "target_modules", ["in_proj", "out_proj", "x_proj", "dt_proj"]
            )
            peft_config = LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=target_modules, bias="none",
            )
            self.model = inject_adapter_in_model(peft_config, self.model)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"[lm_adapters] LoRA loaded: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys")
        self.model.to(self.device)
