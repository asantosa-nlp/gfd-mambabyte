"""gfd/asr_adapters.py

Whisper adapter exposing the interface consumed by ByteLevelFusionDecoder.
Adapted from blt-gfd-jvsu-asr/src/fusion/asr_adapters.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class WhisperState:
    encoder_outputs: Any
    forced_prefix: list[int]


@dataclass
class WhisperPrefixNode:
    asr_tokens: tuple[int, ...]
    past_key_values: Any
    next_logits: torch.Tensor
    children: dict[int, "WhisperPrefixNode"] = field(default_factory=dict)


# Whisper language tag mapping — Whisper uses non-standard codes for some languages
WHISPER_LANG_CODES = {
    "jv": "jw",   # Javanese
    "jav": "jw",
    "su": "su",   # Sundanese
    "sun": "su",
    "id": "id",   # Indonesian
}


class WhisperAdapter:
    def __init__(
        self,
        model_name_or_path: str,
        language_code: str = "id",
        device: str = "cuda",
        dtype=torch.bfloat16,
        initial_prompt: str | None = None,
    ):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.initial_prompt = initial_prompt
        try:
            self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        except (TypeError, OSError):
            # Fine-tuned checkpoints often omit vocab.json; fall back to base model processor
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.tokenizer = self.processor.tokenizer
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=dtype, attn_implementation="eager"
        ).to(device).eval()
        # Resolve Whisper's internal language code
        self.language_code = WHISPER_LANG_CODES.get(language_code, language_code)

    @torch.inference_mode()
    def encode(self, audio_array) -> WhisperState:
        inputs = self.processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, dtype=next(self.model.parameters()).dtype)
        enc_out = self.model.get_encoder()(inputs)
        try:
            forced = self.processor.get_decoder_prompt_ids(
                language=self.language_code, task="transcribe", no_timestamps=True
            )
            forced_ids = [t for _, t in forced]
            sot = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
            nots = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            forced_prefix = [sot] + forced_ids + [nots]
        except Exception:
            forced_prefix = [self.tokenizer.bos_token_id]
        if self.initial_prompt:
            # Prepend [<|startofprev|>, ...prompt_tokens...] before SOT per Whisper spec
            prompt_ids = self.processor.get_prompt_ids(self.initial_prompt)
            forced_prefix = list(prompt_ids) + forced_prefix
        return WhisperState(encoder_outputs=enc_out, forced_prefix=forced_prefix)

    @torch.inference_mode()
    def init_prefix_cache(self, state: WhisperState) -> WhisperPrefixNode:
        ids_t = torch.tensor([state.forced_prefix], device=self.device, dtype=torch.long)
        out = self.model(
            decoder_input_ids=ids_t,
            encoder_outputs=state.encoder_outputs,
            use_cache=True,
            return_dict=True,
        )
        return WhisperPrefixNode(
            asr_tokens=tuple(),
            past_key_values=out.past_key_values,
            next_logits=out.logits[0, -1],
        )

    @torch.inference_mode()
    def extend_prefix_cache(
        self, state: WhisperState, prefix_node: WhisperPrefixNode, next_token_id: int
    ) -> tuple[WhisperPrefixNode, bool]:
        cached = prefix_node.children.get(int(next_token_id))
        if cached is not None:
            return cached, True
        ids_t = torch.tensor([[int(next_token_id)]], device=self.device, dtype=torch.long)
        out = self.model(
            decoder_input_ids=ids_t,
            encoder_outputs=state.encoder_outputs,
            past_key_values=prefix_node.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        child = WhisperPrefixNode(
            asr_tokens=prefix_node.asr_tokens + (int(next_token_id),),
            past_key_values=out.past_key_values,
            next_logits=out.logits[0, -1],
        )
        prefix_node.children[int(next_token_id)] = child
        return child, False

    @torch.inference_mode()
    def next_token_logits_cached(self, state, asr_tokens, prefix_node):
        if prefix_node is None or tuple(asr_tokens) != prefix_node.asr_tokens:
            return self.next_token_logits(state, asr_tokens)
        logits = prefix_node.next_logits
        eos_id = self.tokenizer.eos_token_id
        eos_score = float(torch.log_softmax(logits, dim=-1)[eos_id].item())
        return logits, eos_score, False

    @torch.inference_mode()
    def next_token_logits(self, state: WhisperState, asr_tokens: list[int]):
        ids = state.forced_prefix + asr_tokens
        # Whisper positional embedding has max_target_positions slots (448 for medium).
        # Truncate from the left (drop oldest asr_tokens) to stay within the limit.
        max_pos = getattr(self.model.config, "max_target_positions", 448)
        if len(ids) > max_pos:
            ids = ids[-max_pos:]
        ids_t = torch.tensor([ids], device=self.device, dtype=torch.long)
        out = self.model(
            decoder_input_ids=ids_t,
            encoder_outputs=state.encoder_outputs,
            use_cache=False,
        )
        logits = out.logits[0, -1]
        eos_id = self.tokenizer.eos_token_id
        eos_score = float(torch.log_softmax(logits, dim=-1)[eos_id].item())
        return logits, eos_score, False
