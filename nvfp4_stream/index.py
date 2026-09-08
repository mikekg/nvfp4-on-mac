# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Read safetensors metadata without touching tensor payloads."""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path


DTYPE_SIZE = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

PROJECTIONS = ("up_proj", "down_proj")
SOURCE_PARTS = ("weight", "weight_scale", "weight_scale_2", "input_scale")
NVFP4_SCALE_DENOM = 6.0 * 448.0


@dataclass(frozen=True)
class TensorLoc:
    path: str
    start: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    name: str


class ModelOptIndex:
    """Byte-range index for an unmodified sharded ModelOpt checkpoint."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        config_path = self.model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing {config_path}")
        self.config = json.loads(config_path.read_text())
        if self.config.get("model_type") != "nemotron_h":
            raise ValueError("this adapter only supports model_type=nemotron_h")
        hf_quant_path = self.model_dir / "hf_quant_config.json"
        hf_quant = (
            json.loads(hf_quant_path.read_text()) if hf_quant_path.is_file() else {}
        )
        quant_algo = (
            hf_quant.get("quantization", {}).get("quant_algo")
            or (self.config.get("quantization_config") or {}).get("quant_algo")
            or ""
        ).upper()
        if quant_algo not in {"NVFP4", "MIXED_PRECISION", "FP8"}:
            raise ValueError(
                f"unsupported quantization algorithm: {quant_algo or 'none'}"
            )
        files = sorted(glob.glob(str(self.model_dir / "model*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no model*.safetensors under {self.model_dir}")
        shard_matches = [
            re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", Path(name).name)
            for name in files
        ]
        if all(shard_matches):
            totals = {int(match.group(2)) for match in shard_matches}
            if len(totals) != 1:
                raise ValueError("checkpoint shard names disagree on shard count")
            expected = totals.pop()
            present = {int(match.group(1)) for match in shard_matches}
            missing = sorted(set(range(1, expected + 1)) - present)
            if missing:
                preview = ", ".join(str(item) for item in missing[:8])
                suffix = "..." if len(missing) > 8 else ""
                raise FileNotFoundError(
                    f"checkpoint incomplete: found {len(present)}/{expected} shards; "
                    f"missing {preview}{suffix}"
                )

        self.tensors: dict[str, TensorLoc] = {}
        for filename in files:
            self._read_header(filename)
        self.source_bytes = sum(tensor.nbytes for tensor in self.tensors.values())

        pattern = self.config.get("hybrid_override_pattern") or self.config.get(
            "layers_block_type"
        )
        if not pattern:
            raise ValueError("config has no layer pattern")
        self.num_layers = len(pattern)
        self.moe_layers = tuple(
            i for i, kind in enumerate(pattern) if kind in ("E", "moe")
        )
        self.num_experts = (
            int(self.config["n_routed_experts"]) if self.moe_layers else 0
        )
        self.top_k = int(self.config["num_experts_per_tok"]) if self.moe_layers else 0

    def _read_header(self, filename: str) -> None:
        file_size = os.path.getsize(filename)
        with open(filename, "rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors file: {filename}")
            (header_size,) = struct.unpack("<Q", raw)
            header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        for name, item in header.items():
            if name == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            dtype = item["dtype"]
            shape = tuple(item["shape"])
            expected = prod(shape) * DTYPE_SIZE[dtype]
            if end - begin != expected or data_start + end > file_size:
                raise ValueError(f"invalid tensor range for {name} in {filename}")
            if name in self.tensors:
                raise ValueError(f"duplicate tensor {name}")
            self.tensors[name] = TensorLoc(
                filename, data_start + begin, end - begin, dtype, shape, name
            )

    @staticmethod
    def expert_prefix(layer: int, expert: int, projection: str) -> str:
        return f"backbone.layers.{layer}.mixer.experts.{expert}.{projection}"

    def expert_parts(
        self, layer: int, expert: int, projection: str
    ) -> dict[str, TensorLoc]:
        prefix = self.expert_prefix(layer, expert, projection)
        return {part: self.tensors[f"{prefix}.{part}"] for part in SOURCE_PARTS}

    def expert_bytes(self, layer: int) -> int:
        return sum(
            loc.nbytes
            for projection in PROJECTIONS
            for loc in self.expert_parts(layer, 0, projection).values()
        )

    def expert_slot_bytes(self, layer: int) -> int:
        """Bytes kept per expert slot; activation scales are shared per projection."""
        return sum(
            loc.nbytes
            for projection in PROJECTIONS
            for part, loc in self.expert_parts(layer, 0, projection).items()
            if part != "input_scale"
        )

    def validate_experts(self) -> dict[str, int]:
        total = 0
        for layer in self.moe_layers:
            first_bytes = self.expert_bytes(layer)
            for expert in range(self.num_experts):
                size = 0
                for projection in PROJECTIONS:
                    parts = self.expert_parts(layer, expert, projection)
                    weight = parts["weight"]
                    scales = parts["weight_scale"]
                    scale2 = parts["weight_scale_2"]
                    input_scale = parts["input_scale"]
                    if weight.dtype != "U8":
                        raise ValueError(f"{weight.name}: expected U8, got {weight.dtype}")
                    if scales.dtype not in ("F8_E4M3", "F8_E4M3FN", "U8"):
                        raise ValueError(
                            f"{scales.name}: expected E4M3 bytes, got {scales.dtype}"
                        )
                    if scale2.dtype != "F32" or scale2.nbytes != 4:
                        raise ValueError(f"{scale2.name}: expected one F32 scale")
                    if input_scale.dtype != "F32" or input_scale.nbytes != 4:
                        raise ValueError(
                            f"{input_scale.name}: expected one F32 input scale"
                        )
                    if weight.shape[-1] % 4:
                        raise ValueError(f"{weight.name}: cannot view last axis as U32")
                    size += sum(loc.nbytes for loc in parts.values())
                if size != first_bytes:
                    raise ValueError(
                        f"layer {layer} expert {expert}: {size} bytes, expected {first_bytes}"
                    )
            total += first_bytes * self.num_experts
        return {
            "layers": len(self.moe_layers),
            "experts_per_layer": self.num_experts,
            "top_k": self.top_k,
            "source_bytes": self.source_bytes,
            "routed_bytes": total,
            "bytes_per_slot_set": sum(
                self.expert_slot_bytes(layer) for layer in self.moe_layers
            ),
        }
