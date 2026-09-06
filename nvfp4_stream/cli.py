# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .index import ModelOptIndex


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run original ModelOpt NVFP4 Nemotron-H shards on Apple Silicon"
    )
    parser.add_argument("--model", required=True, help="local Hugging Face checkpoint directory")
    parser.add_argument("--prompt", default="Hello", help="one user message")
    parser.add_argument("--output", type=Path, help="write generated text to this file")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--expert-budget-gib", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("metal", "cpu"), default="metal")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mlx-cache-gib", type=float, default=0.5)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit("--model must be a downloaded local directory")

    try:
        index = ModelOptIndex(model_dir)
        summary = index.validate_experts()
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    summary["routed_gib"] = round(summary["routed_bytes"] / (1 << 30), 3)
    summary["slots_per_layer"] = min(
        index.num_experts,
        int(args.expert_budget_gib * (1 << 30)) // summary["bytes_per_slot_set"],
    )
    print(json.dumps(summary, indent=2))
    if args.check_only:
        return

    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from .runtime import aggregate_stats, streaming_model

    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    mx.set_cache_limit(int(args.mlx_cache_gib * (1 << 30)))
    started = time.perf_counter()
    with streaming_model(
        model_dir,
        expert_budget_gib=args.expert_budget_gib,
        workers=args.workers,
        trust_remote_code=args.trust_remote_code,
    ) as (model, tokenizer, pools, reader):
        load_seconds = time.perf_counter() - started
        prompt = args.prompt
        if not args.raw_prompt and tokenizer.has_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=False,
            )
        sampler = make_sampler(temp=args.temp, top_p=args.top_p)
        prefill_step_size = max(
            1,
            min(pool.slots for pool in pools.values()) // index.top_k,
        )
        output = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            prefill_step_size=prefill_step_size,
        ):
            output.append(response.text)
            print(response.text, end="", flush=True)
        print()
        if args.output:
            args.output.write_text("".join(output))
        stats = aggregate_stats(pools, reader)
        stats.update(
            prompt_tokens=response.prompt_tokens,
            prompt_tps=response.prompt_tps,
            generation_tokens=response.generation_tokens,
            generation_tps=response.generation_tps,
            peak_memory_gb=response.peak_memory,
            finish_reason=response.finish_reason,
            load_seconds=load_seconds,
            total_seconds=time.perf_counter() - started,
        )
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
