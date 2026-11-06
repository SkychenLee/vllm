# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling
from vllm.model_executor.models.mistral3 import Mistral3ProcessingInfo
from vllm.model_executor.models.mistral4 import (
    prepare_mistral4_config,
    unpack_mistral4_expert_weights,
)
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.configs.mistral import _remap_mistral_yarn_args

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("attention_factor", [None, 1.25])
def test_mistral4_rope_matches_transformers(attention_factor):
    """The 64 rotary coordinates retain HF frequencies and magnitude."""
    rope = {
        "rope_type": "yarn",
        "factor": 2.0,
        "rope_theta": 10000.0,
        "original_max_position_embeddings": 32,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "llama_4_scaling_beta": 0.1,
    }
    if attention_factor is not None:
        # Explicit HF magnitude must override, not multiply, the mscale ratio.
        rope.update(attention_factor=attention_factor, mscale=2.0)
    config = Mistral4Config(max_position_embeddings=64, rope_parameters=rope)
    original = dict(config.rope_parameters)
    positions = torch.arange(64).unsqueeze(0)
    hf_cos, hf_sin = Mistral4RotaryEmbedding(config)(
        torch.empty(1, 64, config.qk_rope_head_dim), positions
    )
    prepare_mistral4_config(
        SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
    )
    assert config.rope_parameters["apply_yarn_scaling"] is False
    assert "apply_yarn_scaling" not in original
    config.rope_parameters["rope_type"] = "deepseek_llama_scaling"
    with set_current_vllm_config(VllmConfig()):
        actual = get_rope(
            config.qk_rope_head_dim,
            max_position=64,
            is_neox_style=False,
            rope_parameters=config.rope_parameters,
            dtype=torch.float32,
        )
    actual_cos, actual_sin = actual.get_cos_sin(64)
    rotary_half = config.qk_rope_head_dim // 2
    assert actual.rotary_dim == config.qk_rope_head_dim
    torch.testing.assert_close(actual_cos, hf_cos[0, :, :rotary_half])
    torch.testing.assert_close(actual_sin, hf_sin[0, :, :rotary_half])


def test_mistral4_query_scale_crosses_original_context_boundary():
    positions = torch.tensor([0, 8191, 8192, 16384])
    actual = _get_llama_4_scaling(8192, 0.1, positions).flatten()
    expected = 1 + 0.1 * torch.log(torch.tensor([1.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("apply_scale", [False, True])
def test_native_mistral_yarn_preserves_attention_scale_policy(apply_scale):
    mapped = _remap_mistral_yarn_args(
        {"yarn": {"factor": 128, "apply_scale": apply_scale}}
    )["rope_parameters"]
    assert mapped.get("apply_yarn_scaling", True) is apply_scale
    if not apply_scale:
        assert mapped["attention_factor"] == 1.0


@pytest.mark.parametrize(
    "overrides", [{"rope_interleave": False}, {"n_group": 2}, {"topk_group": 2}]
)
def test_mistral4_rejects_unsupported_attention_or_routing(overrides):
    config = SimpleNamespace(rope_parameters={}, **overrides)
    with pytest.raises(NotImplementedError):
        prepare_mistral4_config(
            SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
        )


def test_packed_expert_weights_and_scales_keep_expert_pairing():
    gate_up = torch.arange(48).reshape(2, 6, 4)
    down = torch.arange(24).reshape(2, 4, 3)
    scales = torch.tensor([0.25, 2.0]).reshape(2, 1, 1)
    weights = [
        ("layers.0.mlp.experts.gate_up_proj", gate_up),
        ("layers.0.mlp.experts.gate_up_proj_scale_inv", scales),
        ("layers.0.mlp.experts.down_proj", down),
        ("layers.0.mlp.experts.down_proj_scale_inv", scales),
        ("layers.0.mlp.experts.gate_up_proj_activation_scale", scales.flatten()),
    ]
    expanded = dict(unpack_mistral4_expert_weights(weights, 2))
    for expert in range(2):
        prefix = f"layers.0.mlp.experts.{expert}"
        for projection, expected in (
            ("gate_proj", gate_up[expert, :3]),
            ("up_proj", gate_up[expert, 3:]),
            ("down_proj", down[expert]),
        ):
            value = expanded[f"{prefix}.{projection}.weight"]
            torch.testing.assert_close(value, expected)
            assert (
                value.untyped_storage().data_ptr()
                == expected.untyped_storage().data_ptr()
            )
            torch.testing.assert_close(
                expanded[f"{prefix}.{projection}.weight_scale"], scales[expert]
            )
        assert expanded[f"{prefix}.gate_proj.input_scale"] == scales[expert].item()
    # The Ascend loader expands before conversion; the core loader sees it again.
    twice = dict(unpack_mistral4_expert_weights(expanded.items(), 2))
    assert twice.keys() == expanded.keys()
    assert all(twice[name] is value for name, value in expanded.items())


@pytest.mark.parametrize(
    ("suffix", "shape"),
    [("", (2, 5, 4)), ("", (1, 6, 4)), ("_scale_inv", (2, 3, 1))],
)
def test_packed_expert_shape_mismatch_fails_before_loading(suffix, shape):
    with pytest.raises(ValueError):
        list(
            unpack_mistral4_expert_weights(
                [(f"layers.0.mlp.experts.gate_up_proj{suffix}", torch.empty(shape))],
                2,
            )
        )


def test_mistral4_processor_uses_checkpoint_tokenizer_for_image_expansion():
    tokenizer = object.__new__(MistralTokenizer)
    tokenizer.version = 15
    ctx = MagicMock()
    ctx.get_tokenizer.return_value = tokenizer
    ctx.get_merged_mm_kwargs.return_value = {
        "size": {"longest_edge": 1540},
        "tokenizer": tokenizer,
    }
    with patch(
        "vllm.model_executor.models.mistral3.cached_processor_from_config"
    ) as loader:
        assert Mistral3ProcessingInfo(ctx).get_hf_processor() is loader.return_value
    loader.assert_called_once_with(
        ctx.model_config, processor_cls=PixtralProcessor, size={"longest_edge": 1540}
    )


def test_eagle_dense_draft_initializes_inherited_loader_state():
    """The draft bypasses DeepseekV2Model.__init__, but reuses its loader."""
    from vllm.model_executor.models import deepseek_v2, mistral_large_3_eagle

    config = SimpleNamespace(
        hidden_size=4,
        vocab_size=8,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        rms_norm_eps=1e-6,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        model_type="deepseek_v3",
        n_routed_experts=1,
        n_shared_experts=1,
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config),
        quant_config=None,
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    with (
        patch.object(
            mistral_large_3_eagle,
            "get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch.object(
            mistral_large_3_eagle,
            "VocabParallelEmbedding",
            side_effect=lambda n, d, **kw: torch.nn.Embedding(n, d),
        ),
        patch.object(
            mistral_large_3_eagle,
            "DeepseekV2DecoderLayer",
            side_effect=lambda **kw: torch.nn.Identity(),
        ),
        patch.object(
            mistral_large_3_eagle,
            "RowParallelLinear",
            side_effect=lambda i, o, **kw: torch.nn.Linear(i, o, bias=False),
        ),
        patch.object(
            mistral_large_3_eagle,
            "RMSNorm",
            side_effect=lambda d, **kw: torch.nn.LayerNorm(d),
        ),
    ):
        model = mistral_large_3_eagle.EagleMistralLarge3Model(
            vllm_config=vc, start_layer_id=36
        )
    assert model.is_fused_shared_expert_enabled is False
    assert model.hidden_size == 4
    assert model.config.first_k_dense_replace == 38
    expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    with (
        patch.object(
            deepseek_v2, "fused_moe_make_expert_params_mapping", return_value=[]
        ),
        patch.object(deepseek_v2, "get_pp_missing_layer_names", return_value=[]),
        patch.object(
            deepseek_v2, "get_spec_layer_idx_from_weight_name", return_value=None
        ),
    ):
        loaded = model.load_weights([("fc.weight", expected)])
    assert loaded == {"fc.weight"}
    torch.testing.assert_close(model.fc.weight, expected)
