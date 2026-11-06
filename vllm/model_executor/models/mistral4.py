# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HF-format Mistral 4, using the shared DeepSeek attention and MoE layers."""

from collections.abc import Iterable, Iterator

import torch

from vllm.config import VllmConfig

from .deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2Model


def prepare_mistral4_config(vllm_config: VllmConfig) -> None:
    """Preserve Mistral's RoPE and routing semantics in the shared layers."""
    config = vllm_config.model_config.hf_config
    if not getattr(config, "rope_interleave", True):
        raise NotImplementedError("Mistral 4 requires rope_interleave=True.")
    if getattr(config, "n_group", 1) != 1 or getattr(config, "topk_group", 1) != 1:
        raise NotImplementedError("Mistral 4 grouped expert routing is unsupported.")

    rope_parameters = dict(config.rope_parameters)
    config.rope_parameters = rope_parameters
    # Mistral scales the rotary coordinates, without DeepSeek's additional
    # mscale_all_dim**2 multiplier on the full attention score.
    rope_parameters["apply_yarn_scaling"] = False
    # get_rope already receives qk_rope_head_dim, not the full QK head size.
    rope_parameters["partial_rotary_factor"] = 1.0
    if getattr(config, "llama_4_scaling", None) is None:
        config.llama_4_scaling = {
            "original_max_position_embeddings": rope_parameters[
                "original_max_position_embeddings"
            ],
            "beta": rope_parameters["llama_4_scaling_beta"],
        }


def unpack_mistral4_expert_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    num_experts: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield expert weight/scale views before TP sharding or dequantization.

    Already expanded names pass through unchanged. Keeping FP8 weights as
    views allows a backend to convert one expert at a time instead of allocating
    a BF16 copy of the complete packed expert tensor.
    """
    suffixes = {
        "": "weight",
        "_scale_inv": "weight_scale",
        "_activation_scale": "input_scale",
    }
    for name, tensor in weights:
        matched = False
        for projection in ("gate_up_proj", "down_proj"):
            for suffix, parameter in suffixes.items():
                ending = f".experts.{projection}{suffix}"
                if not name.endswith(ending):
                    continue
                matched = True
                if tensor.ndim == 0 or tensor.shape[0] != num_experts:
                    raise ValueError(
                        f"Unexpected Mistral 4 packed expert shape: {name} "
                        f"{tuple(tensor.shape)}, expected {num_experts} experts"
                    )
                if parameter == "weight":
                    if tensor.ndim != 3:
                        raise ValueError(f"Expected a 3D expert weight: {name}")
                    if projection == "gate_up_proj" and tensor.shape[1] % 2:
                        raise ValueError(f"Packed gate/up size must be even: {name}")
                elif parameter == "weight_scale":
                    if tensor.shape[1:] != (1, 1):
                        raise ValueError(f"Expected per-expert scalar scales: {name}")
                elif tensor.ndim != 1:
                    raise ValueError(f"Expected per-expert activation scales: {name}")

                prefix = name[: -len(ending)]
                for expert_id in range(num_experts):
                    expert_prefix = f"{prefix}.experts.{expert_id}"
                    value = tensor[expert_id]
                    if projection == "gate_up_proj":
                        if parameter == "weight":
                            gate, up = value.chunk(2, dim=0)
                        else:
                            gate = up = value
                        yield f"{expert_prefix}.gate_proj.{parameter}", gate
                        yield f"{expert_prefix}.up_proj.{parameter}", up
                    else:
                        yield f"{expert_prefix}.down_proj.{parameter}", value
                break
            if matched:
                break
        if not matched:
            yield name, tensor


class Mistral4Model(DeepseekV2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        prepare_mistral4_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.quant_config = vllm_config.quant_config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            unpack_mistral4_expert_weights(weights, self.config.n_routed_experts)
        )


class Mistral4ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = Mistral4Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        prepare_mistral4_config(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
