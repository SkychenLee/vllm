# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.model_executor.models.mistral3 import Mistral3ForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper

lora_lst = ["baichuan7B", "baichuan7B-zero", "baichuan7B-zero-regex", "chatglm3-6b"]
BAICHUAN_LORA_MODULES = [
    "W_pack",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]

MOCK_PACKED_MAPPING = {
    "W_pack": ["W_pack"],
    "gate_up_proj": [
        "gate_proj",
        "up_proj",
    ],
}


@pytest.mark.parametrize("lora_name", lora_lst)
def test_load_checkpoints(
    lora_name,
    baichuan_lora_files,
    baichuan_zero_lora_files,
    baichuan_regex_lora_files,
    chatglm3_lora_files,
):
    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in MOCK_PACKED_MAPPING:
            expected_lora_lst.extend(MOCK_PACKED_MAPPING[module])
        else:
            expected_lora_lst.append(module)
    expected_lora_modules = set(expected_lora_lst)
    if lora_name == "baichuan7B":
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_lora_files, max_position_embeddings=4096
        )
        # For the baichuan7B model, load it's LoRA,
        # and the test should pass.
        LoRAModel.from_local_checkpoint(
            baichuan_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    elif lora_name == "baichuan7B-zero":
        # Test that the target_modules contain prefix
        # such as "model.layers.0.self_atten.W_pack", and
        # the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_zero_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_zero_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    elif lora_name == "baichuan7B-zero-regex":
        # Test that the `target_modules` in the form of regular expressions,
        # such as `model\\..*(W_pack|o_proj)`, and the test should pass.
        peft_helper = PEFTHelper.from_local_dir(
            baichuan_regex_lora_files, max_position_embeddings=4096
        )
        LoRAModel.from_local_checkpoint(
            baichuan_regex_lora_files,
            expected_lora_modules,
            peft_helper=peft_helper,
            lora_model_id=1,
            device="cpu",
            model_vocab_size=64000,
        )
    else:
        # For the baichuan7B model, load chatglm3-6b's LoRA,
        # and the test should raise the following error.
        expected_error = "Please verify that the loaded LoRA module is correct"  # noqa: E501
        peft_helper = PEFTHelper.from_local_dir(
            chatglm3_lora_files, max_position_embeddings=4096
        )
        with pytest.raises(ValueError, match=expected_error):
            LoRAModel.from_local_checkpoint(
                chatglm3_lora_files,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=1,
                device="cpu",
                model_vocab_size=64000,
            )


def test_lora_weights_mapping(baichuan_lora_files):
    expected_lora_lst: list[str] = []
    for module in BAICHUAN_LORA_MODULES:
        if module in MOCK_PACKED_MAPPING:
            expected_lora_lst.extend(MOCK_PACKED_MAPPING[module])
        else:
            expected_lora_lst.append(module)
    expected_lora_modules = set(expected_lora_lst)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
        },
        orig_to_new_substr={
            ".layers.": ".baichuan_layers.",
        },
    )
    peft_helper = PEFTHelper.from_local_dir(
        baichuan_lora_files, max_position_embeddings=4096
    )
    lora_model = LoRAModel.from_local_checkpoint(
        baichuan_lora_files,
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        model_vocab_size=64000,
        weights_mapper=hf_to_vllm_mapper,
    )
    for name in lora_model.loras:
        assert name.startswith(hf_to_vllm_mapper.orig_to_new_prefix["model."])
        assert ".baichuan_layers." in name


def test_gemma4_lora_weights_mapping():
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = "base_model.model.model.language_model.layers.9.mlp.down_proj.lora_A.weight"
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.mlp.down_proj",
        True,
    )


def test_gemma4_moe_lora_weights_mapping():
    mapper = Gemma4ForCausalLM.hf_to_vllm_mapper
    name = (
        "base_model.model.model.language_model.layers.9.moe.experts."
        "gate_up_proj.lora_B.weight"
    )
    assert parse_fine_tuned_lora_name(name, mapper) == (
        "model.layers.9.moe.gate_up_proj",
        False,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("file_format", ["safetensors", "bin"])
@pytest.mark.parametrize("adapter_suffix", ["", ".default"])
@pytest.mark.parametrize(
    "prefix", ["language_model.model.", "base_model.model.model.language_model."]
)
def test_load_shared_expert_lora_with_default_adapter_name(
    tmp_path, file_format, adapter_suffix, prefix
):
    """Load the reported shared-expert keys without changing tensors or scaling."""
    targets = {"gate_proj", "up_proj", "down_proj"}
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4, "target_modules": sorted(targets)})
    )
    a = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    b = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    tensors = {
        f"{prefix}layers.0.mlp.shared_experts.{target}.lora_{side}"
        f"{adapter_suffix}.weight": tensor.clone()
        for target in targets
        for side, tensor in (("A", a), ("B", b))
    }
    checkpoint = tmp_path / f"adapter_model.{file_format}"
    if file_format == "safetensors":
        save_file(tensors, str(checkpoint))
    else:
        torch.save(tensors, checkpoint)

    peft_helper = PEFTHelper.from_local_dir(str(tmp_path), max_position_embeddings=None)
    model = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        expected_lora_modules=targets,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        dtype=torch.bfloat16,
        weights_mapper=Mistral3ForConditionalGeneration.hf_to_vllm_mapper,
    )
    assert len(model.loras) == 3
    for target in targets:
        lora = model.loras[f"language_model.model.layers.0.mlp.shared_experts.{target}"]
        assert lora.scaling == 2
        torch.testing.assert_close(lora.lora_a, a.to(torch.bfloat16))
        torch.testing.assert_close(lora.lora_b, b.to(torch.bfloat16))


@pytest.mark.cpu_test
@pytest.mark.parametrize("side", ["A", "B"])
def test_default_adapter_aliases_cannot_overwrite_lora_weights(side):
    module = "language_model.model.layers.0.mlp.shared_experts.down_proj"
    tensors = {
        f"{module}.lora_{side}.weight": torch.ones(2, 2),
        f"{module}.lora_{side}.default.weight": torch.zeros(2, 2),
    }
    peft_helper = PEFTHelper(r=2, lora_alpha=4, target_modules=["down_proj"])
    with pytest.raises(ValueError, match="Duplicate LoRA weight"):
        LoRAModel.from_lora_tensors(1, tensors, peft_helper, device="cpu")
