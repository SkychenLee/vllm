# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy out-of-window frees and single-batch KV admission bounds."""

import pytest
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import ChunkedLocalAttentionSpec, SlidingWindowSpec
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler, mock_kv

NUM_PROMPT_TOKENS = 100
BLOCK_SIZE = 16
SLIDING_WINDOW = 16
# Tokens 0..84 are outside the window of the next token to compute
# (100 - 16 + 1 = 85), i.e. 5 full blocks.
NUM_OUT_OF_WINDOW_BLOCKS = 85 // BLOCK_SIZE
# Chunked-local skips whole chunks left of the current one:
# (100 // 32) * 32 = 96 computed tokens -> 6 full blocks.
CHUNK_SIZE = 32
NUM_OUT_OF_CHUNK_BLOCKS = (NUM_PROMPT_TOKENS // CHUNK_SIZE) * CHUNK_SIZE // BLOCK_SIZE


def _make_model_runner_output(
    scheduler_output: SchedulerOutput,
    token_id: int = 0,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _create_swa_scheduler(async_scheduling: bool):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )


def _create_chunked_scheduler(async_scheduling: bool):
    return create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=async_scheduling,
        kv_cache_spec=ChunkedLocalAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            attention_chunk_size=CHUNK_SIZE,
        ),
    )


def _num_null_blocks(scheduler, request_id: str) -> int:
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    null_block = manager._null_block
    return sum(1 for b in manager.req_to_blocks[request_id] if b is null_block)


def test_num_in_flight_tokens_accounting():
    scheduler = create_scheduler(async_scheduling=True)
    request = create_requests(num_requests=1, num_tokens=NUM_PROMPT_TOKENS)[0]
    scheduler.add_request(request)

    out0 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS
    # Async: decode scheduled before the prefill output is processed.
    out1 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1

    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert request.num_in_flight_tokens == 1
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    assert request.num_in_flight_tokens == 0


def test_swa_free_uses_optimistic_computed_tokens():
    """Async frees use scheduled tokens even before prefill output arrives."""
    scheduler = _create_swa_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id
    block_pool = scheduler.kv_cache_manager.block_pool

    out0 = scheduler.schedule()  # prefill, in flight from here on
    free_after_prefill = block_pool.get_num_free_blocks()

    out1 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS
    assert (
        block_pool.get_num_free_blocks()
        == free_after_prefill + NUM_OUT_OF_WINDOW_BLOCKS
    )
    # Processing the outstanding outputs must not free those blocks again.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler.schedule()
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS
    assert (
        block_pool.get_num_free_blocks()
        == free_after_prefill + NUM_OUT_OF_WINDOW_BLOCKS
    )


def test_swa_free_immediate_when_sync():
    """Sync: no in-flight step at schedule time, frees happen at the first
    decode allocation as before."""
    scheduler = _create_swa_scheduler(async_scheduling=False)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id

    out0 = scheduler.schedule()
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    assert request.num_in_flight_tokens == 0

    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS


def test_chunked_local_free_uses_optimistic_computed_tokens():
    """Chunked-local frees also use the optimistic computed-token count."""
    scheduler = _create_chunked_scheduler(async_scheduling=True)
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id
    block_pool = scheduler.kv_cache_manager.block_pool

    out0 = scheduler.schedule()  # prefill, in flight from here on
    free_after_prefill = block_pool.get_num_free_blocks()

    out1 = scheduler.schedule()
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS + 1
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_CHUNK_BLOCKS
    assert (
        block_pool.get_num_free_blocks() == free_after_prefill + NUM_OUT_OF_CHUNK_BLOCKS
    )
    # Processing the outstanding outputs must not free those blocks again.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler.schedule()
    scheduler.update_from_output(out1, _make_model_runner_output(out1))
    scheduler.schedule()
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_CHUNK_BLOCKS
    assert (
        block_pool.get_num_free_blocks() == free_after_prefill + NUM_OUT_OF_CHUNK_BLOCKS
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    ("spec", "expected_blocks"),
    [
        (
            SlidingWindowSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=128,
            ),
            17,  # ceil((128 - 1 + 128) / 16) + 1
        ),
        (
            ChunkedLocalAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                attention_chunk_size=128,
            ),
            16,  # ceil((128 + 128) / 16)
        ),
    ],
    ids=["sliding_window", "chunked_local"],
)
def test_startup_and_runtime_admission_use_single_batch(
    spec, expected_blocks: int, async_scheduling: bool
):
    """A pool sized for one batch admits long prompts with either scheduler."""
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=128,
        max_model_len=2048,
        num_blocks=expected_blocks + 1,  # Include the null block.
        async_scheduling=async_scheduling,
        kv_cache_spec=spec,
    )
    assert (
        spec.max_memory_usage_bytes(scheduler.vllm_config)
        == expected_blocks * spec.page_size_bytes
    )
    request = create_requests(num_requests=1, num_tokens=1024, block_size=BLOCK_SIZE)[0]
    assert (
        scheduler.kv_cache_manager.allocate_slots(
            request, 128, full_sequence_must_fit=True
        )
        is not None
    )


def test_connector_finish_frees_on_optimistic_basis():
    """Connector handoff prunes without waiting for prefill output."""
    scheduler = create_scheduler(
        block_size=BLOCK_SIZE,
        async_scheduling=True,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
        kv_cache_spec=SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=SLIDING_WINDOW,
        ),
    )
    request = create_requests(
        num_requests=1, num_tokens=NUM_PROMPT_TOKENS, block_size=BLOCK_SIZE
    )[0]
    scheduler.add_request(request)
    req_id = request.request_id

    out0 = scheduler.schedule()  # prefill, in flight from here on
    assert request.num_in_flight_tokens == NUM_PROMPT_TOKENS
    assert _num_null_blocks(scheduler, req_id) == 0

    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS

    # Pruning after output processing remains idempotent.
    scheduler.update_from_output(out0, _make_model_runner_output(out0))
    scheduler._connector_finished(request)
    assert _num_null_blocks(scheduler, req_id) == NUM_OUT_OF_WINDOW_BLOCKS
