# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Numerical smoke test for streamed and resident NVFP4 inference."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from nvfp4_stream.index import NVFP4_SCALE_DENOM, ModelOptIndex
from nvfp4_stream.reader import ExpertReader
from nvfp4_stream.runtime import (
    ExpertPool,
    _expert_module,
    _static_nvfp4_module,
    aggregate_stats,
)


def run(device) -> None:
    mx.set_default_device(device)
    mx.random.seed(0)
    experts, top_k, dims = 4, 2, 32
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "hybrid_override_pattern": "E",
                    "n_routed_experts": experts,
                    "num_experts_per_tok": top_k,
                }
            )
        )
        (path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "NVFP4"}})
        )

        source = {}
        dense = {}
        quantized = {}
        input_global_scales = {
            "up_proj": mx.array(4.0, dtype=mx.float32),
            "down_proj": mx.array(64.0, dtype=mx.float32),
        }
        for expert in range(experts):
            dense[expert] = {}
            for projection in ("up_proj", "down_proj"):
                weight = mx.random.normal((dims, dims), dtype=mx.float32)
                global_scale = mx.max(mx.abs(weight)).astype(mx.float32)
                qweight, scales = mx.quantize(
                    weight,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=global_scale,
                )
                dequant = mx.dequantize(
                    qweight,
                    scales,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=global_scale,
                    dtype=mx.bfloat16,
                )
                mx.eval(qweight, scales, global_scale, dequant)
                quantized[(expert, projection)] = (qweight, scales, global_scale)
                prefix = f"backbone.layers.0.mixer.experts.{expert}.{projection}"
                q_u8 = np.asarray(qweight).view(np.uint8).reshape(dims, dims // 2)
                source[f"{prefix}.weight"] = mx.array(q_u8)
                source[f"{prefix}.weight_scale"] = scales
                source[f"{prefix}.weight_scale_2"] = (
                    global_scale / NVFP4_SCALE_DENOM
                ).reshape(1)
                source[f"{prefix}.input_scale"] = (
                    input_global_scales[projection] / NVFP4_SCALE_DENOM
                ).reshape(1)
                dense[expert][projection] = dequant
        mx.save_safetensors(str(path / "model-00001-of-00001.safetensors"), source)

        index = ModelOptIndex(path)
        reader = ExpertReader(index)
        try:
            stats_pool = ExpertPool(
                index, reader, layer=0, slots=2, expert_stats=True
            )
            stats_pool.ensure([0, 1])
            stats_pool.ensure([2])
            stats_pool.ensure([0])
            stats = aggregate_stats({0: stats_pool}, reader, expert_stats=True)
            assert stats["misses"] == 4
            assert stats["cold_misses"] == 3
            assert stats["capacity_misses"] == 1
            assert stats["misses"] == (
                stats["cold_misses"] + stats["capacity_misses"]
            )
            assert stats["unique_experts_accessed"] == 3
            assert stats["experts_loaded"] == 2

            stats_pool.reset_stats()
            reader.reset_stats()
            stats_pool.ensure([1])
            stats = aggregate_stats({0: stats_pool}, reader, expert_stats=True)
            assert stats["misses"] == stats["capacity_misses"] == 1
            assert stats["cold_misses"] == 0
            assert stats["unique_experts_accessed"] == 1

            pool = ExpertPool(
                index, reader, layer=0, slots=experts, expert_stats=True
            )
            module = _expert_module(pool)
            module._forward = mx.compile(module._forward, inputs=pool.tensors)
            x = mx.random.normal((1, 1, dims), dtype=mx.bfloat16)
            indices = mx.array([[[0, 2]]], dtype=mx.uint32)
            streamed = module(x, indices)

            def nvfp4_roundtrip(value, projection):
                global_scale = input_global_scales[projection]
                qvalue, scales = mx.quantize(
                    value,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=global_scale,
                )
                return mx.dequantize(
                    qvalue,
                    scales,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=global_scale,
                    dtype=value.dtype,
                )

            x_quantized = nvfp4_roundtrip(x, "up_proj")
            reference = []
            for expert in (0, 2):
                up = x_quantized @ dense[expert]["up_proj"].T
                activated = nvfp4_roundtrip(nn.relu2(up), "down_proj")
                reference.append(activated @ dense[expert]["down_proj"].T)
            expected = mx.stack(reference, axis=2)
            mx.eval(streamed, expected)
            if not mx.allclose(streamed, expected, rtol=3e-2, atol=1).item():
                raise AssertionError(
                    f"streamed result mismatch: {streamed} != {expected}"
                )

            pool.preload_all()
            pool.reset_stats()
            reader.reset_stats()
            reads = reader.reads
            hits = pool.hits
            resident = module(x, indices)
            mx.eval(resident)
            if reader.reads != reads:
                raise AssertionError("resident forward read checkpoint payload")
            if pool.hits != hits:
                raise AssertionError("resident forward consulted the streamed cache")
            if not mx.allclose(resident, streamed, rtol=0, atol=0).item():
                raise AssertionError(
                    f"resident result differs from streamed: {resident} != {streamed}"
                )
            stats = aggregate_stats({0: pool}, reader, expert_stats=True)
            assert stats["experts_loaded"] == experts
            assert "unique_experts_accessed" not in stats
            assert stats["misses"] == stats["cold_misses"] == 0
            assert stats["capacity_misses"] == 0

            qweight, scales, global_scale = quantized[(0, "up_proj")]
            static = _static_nvfp4_module(qweight.shape, scales.shape, bias=False)
            static.weight = mx.expand_dims(qweight, 0)
            static.scales = mx.expand_dims(scales, 0)
            static.global_scale = global_scale.reshape((1,))
            static.input_global_scale = input_global_scales["up_proj"]
            static_result = static(x)
            static_expected = x_quantized @ dense[0]["up_proj"].T
            mx.eval(static_result, static_expected)
            if not mx.allclose(
                static_result, static_expected, rtol=3e-2, atol=1
            ).item():
                raise AssertionError(
                    "static NVFP4 result mismatch: "
                    f"{static_result} != {static_expected}"
                )
        finally:
            reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("metal", "cpu"), default="metal")
    args = parser.parse_args()
    run(mx.gpu if args.device == "metal" else mx.cpu)
    print(f"{args.device} streamed/resident NVFP4 smoke: PASS")
