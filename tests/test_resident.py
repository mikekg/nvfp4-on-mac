# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Check expert arithmetic, disk loading, eviction, and resident reuse."""

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
    """Run numerical and cache-lifecycle smoke checks on one MLX device."""
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
                    "layers_block_type": ["moe"],
                    "n_routed_experts": experts,
                    "num_experts_per_tok": top_k,
                }
            )
        )
        (path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "MIXED_PRECISION"}})
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
        assert index.expert_formats == {0: {"up_proj": "nvfp4", "down_proj": "nvfp4"}}
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

        prefix = "backbone.layers.0.mixer.experts.1.up_proj"
        for key, replacement in (
            (f"{prefix}.weight", source[f"{prefix}.weight"].reshape(dims // 2, dims)),
            (f"{prefix}.weight_scale", None),
        ):
            invalid = dict(source)
            if replacement is None:
                invalid.pop(key)
            else:
                invalid[key] = replacement
            mx.save_safetensors(str(path / "model-00001-of-00001.safetensors"), invalid)
            try:
                ModelOptIndex(path)
            except ValueError as error:
                assert "expert tensor layouts differ" in str(error)
            else:
                raise AssertionError("invalid expert layout was accepted")

        bf16_path = Path(tmp) / "bf16-moe"
        bf16_path.mkdir()
        (bf16_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "layers_block_type": ["moe"],
                    "n_routed_experts": 2,
                    "num_experts_per_tok": 1,
                }
            )
        )
        bf16_weights = {
            0: {
                "up_proj": mx.array(
                    [[1, 0], [0, 1], [1, 1]], dtype=mx.bfloat16
                ),
                "down_proj": mx.array([[1, 0, 0], [0, 0, 1]], dtype=mx.bfloat16),
            },
            1: {
                "up_proj": mx.array(
                    [[0, -1], [1, 0], [1, -1]], dtype=mx.bfloat16
                ),
                "down_proj": mx.array([[1, 1, 0], [0, 1, -1]], dtype=mx.bfloat16),
            },
        }
        mx.save_safetensors(
            str(bf16_path / "model-00001-of-00001.safetensors"),
            {
                f"backbone.layers.0.mixer.experts.{expert}.{projection}.weight": weight
                for expert, projections in bf16_weights.items()
                for projection, weight in projections.items()
            },
        )

        bf16_index = ModelOptIndex(bf16_path)
        assert bf16_index.expert_formats == {0: {"up_proj": "bf16", "down_proj": "bf16"}}
        assert bf16_index.expert_summary["bytes_per_slot_set"] == 24
        bf16_reader = ExpertReader(bf16_index)
        try:
            x = mx.array([[[2, -1], [2, -1]]], dtype=mx.bfloat16)
            indices = mx.array([[[0], [1]]], dtype=mx.uint32)
            expected = mx.array([[[[4, 1]], [[5, -5]]]], dtype=mx.bfloat16)

            streamed_pool = ExpertPool(
                bf16_index, bf16_reader, layer=0, slots=1, expert_stats=True
            )
            assert streamed_pool.tensors["up_proj"]["weight"].dtype == mx.bfloat16
            streamed_module = _expert_module(streamed_pool)
            streamed_module._forward = mx.compile(
                streamed_module._forward, inputs=streamed_pool.tensors
            )
            streamed = streamed_module(x, indices)
            mx.eval(streamed, expected)
            assert mx.allclose(streamed, expected, rtol=0, atol=0).item()
            assert streamed_pool.misses == 2
            assert streamed_pool.evictions == 1

            resident_pool = ExpertPool(
                bf16_index, bf16_reader, layer=0, slots=2, expert_stats=True
            )
            resident_pool.preload_all()
            reads = bf16_reader.reads
            hits = resident_pool.hits
            resident = _expert_module(resident_pool)(x, indices)
            mx.eval(resident)
            assert bf16_reader.reads == reads
            assert resident_pool.hits == hits
            assert mx.allclose(resident, expected, rtol=0, atol=0).item()
        finally:
            bf16_reader.close()

        dense_path = Path(tmp) / "dense"
        dense_path.mkdir()
        (dense_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "nemotron_h",
                    "hybrid_override_pattern": "M",
                }
            )
        )
        (dense_path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "NVFP4"}})
        )
        mx.save_safetensors(
            str(dense_path / "model-00001-of-00001.safetensors"),
            {"backbone.layers.0.norm.weight": mx.ones((1,))},
        )
        dense_index = ModelOptIndex(dense_path)
        assert dense_index.num_layers == 1 and dense_index.moe_layers == ()
        assert dense_index.num_experts == dense_index.top_k == 0
        assert dense_index.expert_summary["bytes_per_slot_set"] == 0

        (dense_path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "FP8"}})
        )
        assert ModelOptIndex(dense_path).moe_layers == ()

        llama_path = Path(tmp) / "llama"
        llama_path.mkdir()
        (llama_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 32,
                    "quantization_config": {"quant_algo": "NVFP4"},
                }
            )
        )
        mx.save_safetensors(
            str(llama_path / "model-00001-of-00001.safetensors"),
            {"model.embed_tokens.weight": mx.ones((1,))},
        )
        llama_index = ModelOptIndex(llama_path)
        assert llama_index.num_layers == 32 and llama_index.moe_layers == ()

        (llama_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "num_hidden_layers": 32,
                    "torch_dtype": "bfloat16",
                }
            )
        )
        assert ModelOptIndex(llama_path).quant_algo == ""

        (llama_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen2",
                    "num_hidden_layers": 36,
                    "torch_dtype": "bfloat16",
                }
            )
        )
        qwen_index = ModelOptIndex(llama_path)
        assert qwen_index.num_layers == 36 and qwen_index.moe_layers == ()
        assert qwen_index.quant_algo == ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("metal", "cpu"), default="metal")
    args = parser.parse_args()
    run(mx.gpu if args.device == "metal" else mx.cpu)
    print(f"{args.device} streamed/resident expert smoke: PASS")
