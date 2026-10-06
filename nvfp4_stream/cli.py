# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Validate, load, generate, and report through the command-line interface."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .index import ModelOptIndex
from .runtime import routed_mlp, select_expert_mode


def _parser() -> argparse.ArgumentParser:
    """Build the command-line interface for validation and inference."""
    parser = argparse.ArgumentParser(
        description="Run original ModelOpt-layout NVFP4 shards on MLX"
    )
    parser.add_argument("--model", required=True, help="local Hugging Face checkpoint directory")
    parser.add_argument("--prompt", default="Hello", help="one user message")
    parser.add_argument("--output", type=Path, help="write generated text to this file")
    parser.add_argument("--stats-output", type=Path, help="write run metrics as JSON")
    parser.add_argument(
        "--quiet-inference", action="store_true", help="suppress generated text"
    )
    parser.add_argument(
        "--expert-stats",
        action="store_true",
        help="report per-run expert-cache statistics",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--expert-budget-gib", type=float, default=8.0)
    parser.add_argument(
        "--expert-mode", choices=("auto", "resident", "stream"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=("metal", "cpu"), default="metal")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--mlx-cache-gib", type=float, default=0.5)
    parser.add_argument(
        "--compile", action="store_true", help="compile NVFP4 expert compute with MLX"
    )
    parser.add_argument("--runs", type=int, default=1)
    return parser


def main(argv=None) -> None:
    """Validate, load, and run a local checkpoint through the selected MLX path."""
    args = _parser().parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.prefill_chunk is not None and args.prefill_chunk < 1:
        raise SystemExit("--prefill-chunk must be at least 1")
    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise SystemExit("--model must be a downloaded local directory")

    try:
        index = ModelOptIndex(model_dir)
        summary = index.expert_summary
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    expert_mode = select_expert_mode(
        args.expert_mode if index.moe_layers else "resident", index.source_bytes
    )
    compile_experts = args.compile and bool(index.moe_layers)
    summary["expert_mode"] = expert_mode
    summary["format"] = index.quant_algo or index.config.get(
        "torch_dtype", "unknown"
    )
    summary["compile"] = compile_experts
    summary["runs"] = args.runs
    summary["expert_stats"] = args.expert_stats
    summary["source_gib"] = round(index.source_bytes / (1 << 30), 3)
    summary["routed_gib"] = round(summary["routed_bytes"] / (1 << 30), 3)
    summary["slots_per_layer"] = (
        index.num_experts
        if expert_mode == "resident"
        else min(
            index.num_experts,
            int(args.expert_budget_gib * (1 << 30))
            // summary["bytes_per_slot_set"],
        )
    )
    if args.prefill_chunk is not None:
        prefill_chunk = args.prefill_chunk
    elif expert_mode == "resident":
        prefill_chunk = 2048
    else:
        prefill_chunk = max(1, summary["slots_per_layer"] // index.top_k)
    summary["prefill_chunk"] = prefill_chunk
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
        index,
        expert_budget_gib=args.expert_budget_gib,
        expert_mode=expert_mode,
        workers=args.workers,
        trust_remote_code=args.trust_remote_code,
        expert_stats=args.expert_stats,
    ) as (model, tokenizer, pools, reader):
        load_seconds = time.perf_counter() - started
        if compile_experts:
            for layer, pool in pools.items():
                expert = routed_mlp(model, layer).switch_mlp
                expert._forward = mx.compile(
                    expert._forward, inputs=pool.tensors
                )
        prompt = args.prompt
        if not args.raw_prompt and tokenizer.has_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=False,
            )
        sampler = make_sampler(temp=args.temp, top_p=args.top_p)
        run_stats = []
        for run in range(1, args.runs + 1):
            if args.expert_stats:
                reader.reset_stats()
                for pool in pools.values():
                    pool.reset_stats()
            run_started = time.perf_counter()
            output = []
            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                prefill_step_size=prefill_chunk,
            ):
                output.append(response.text)
                if not args.quiet_inference:
                    print(response.text, end="", flush=True)
            if not args.quiet_inference:
                print()
            if args.output:
                output_path = args.output
                if args.runs > 1:
                    output_path = output_path.with_name(
                        f"{output_path.stem}-run{run}{output_path.suffix}"
                    )
                output_path.write_text("".join(output))
            stats = aggregate_stats(pools, reader, args.expert_stats)
            stats.update(
                run=run,
                compile=compile_experts,
                prompt_tokens=response.prompt_tokens,
                prompt_tps=response.prompt_tps,
                generation_tokens=response.generation_tokens,
                generation_tps=response.generation_tps,
                peak_memory_gb=response.peak_memory,
                finish_reason=response.finish_reason,
                load_seconds=load_seconds,
                run_seconds=time.perf_counter() - run_started,
                total_seconds=time.perf_counter() - started,
            )
            run_stats.append(stats)
            print(json.dumps(stats, indent=2))
        if args.stats_output:
            args.stats_output.write_text(json.dumps(run_stats, indent=2))


if __name__ == "__main__":
    main()
