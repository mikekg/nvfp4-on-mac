"""Nemotron-H loader with SSD-paged routed experts."""

from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .index import PROJECTIONS, ModelOptIndex
from .reader import ExpertReader


_SANITIZER_LOCK = threading.Lock()


class ExpertPool:
    """Fixed per-layer slots with protected-step LRU replacement."""

    def __init__(self, index: ModelOptIndex, reader: ExpertReader, layer: int, slots: int):
        import mlx.core as mx

        self.index = index
        self.reader = reader
        self.layer = layer
        self.slots = slots
        self.id_to_slot: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(slots - 1, -1, -1))
        self.slot_table = np.full(index.num_experts, -1, dtype=np.int32)
        self.hits = self.misses = self.evictions = 0
        self.input_global_scales: dict[str, np.float32] = {}
        self.input_global_scale_arrays = {}

        self.tensors = {}
        for projection in PROJECTIONS:
            parts = index.expert_parts(layer, 0, projection)
            weight_shape = parts["weight"].shape
            weight_shape = weight_shape[:-1] + (weight_shape[-1] // 4,)
            self.tensors[projection] = {
                "weight": mx.zeros((slots,) + weight_shape, dtype=mx.uint32),
                "scales": mx.zeros(
                    (slots,) + parts["weight_scale"].shape, dtype=mx.uint8
                ),
                "global_scale": mx.ones((slots,), dtype=mx.float32),
            }

    def evaluate(self) -> None:
        import mlx.core as mx

        tensors = [
            tensor for parts in self.tensors.values() for tensor in parts.values()
        ]
        tensors.extend(self.input_global_scale_arrays.values())
        mx.eval(tensors)

    def _record_input_scale(self, projection: str, value: np.float32) -> None:
        import mlx.core as mx

        previous = self.input_global_scales.get(projection)
        if previous is not None and previous.tobytes() != value.tobytes():
            raise ValueError(
                f"layer {self.layer} {projection}: experts have different input_scale"
            )
        if previous is None:
            self.input_global_scales[projection] = value
            self.input_global_scale_arrays[projection] = mx.array(value, mx.float32)

    def _allocate(self, expert: int, protected: set[int]) -> int:
        if self.free:
            slot = self.free.pop()
        else:
            victim = next((item for item in self.id_to_slot if item not in protected), None)
            if victim is None:
                raise RuntimeError("expert pool has no evictable slot")
            slot = self.id_to_slot.pop(victim)
            self.slot_table[victim] = -1
            self.evictions += 1
        self.id_to_slot[expert] = slot
        self.slot_table[expert] = slot
        return slot

    def ensure(self, experts: list[int]) -> None:
        wanted = set(experts)
        missing = []
        for expert in experts:
            if expert in self.id_to_slot:
                self.id_to_slot.move_to_end(expert)
                self.hits += 1
            else:
                self.misses += 1
                self._allocate(expert, wanted)
                missing.append(expert)
        if not missing:
            return

        rows = self.reader.read_experts(self.layer, missing)
        for expert in missing:
            slot = self.id_to_slot[expert]
            for projection in PROJECTIONS:
                self._record_input_scale(
                    projection,
                    rows[expert][projection]["input_global_scale"],
                )
                for part in ("weight", "scales", "global_scale"):
                    self.tensors[projection][part][slot] = rows[expert][projection][part]
        self.evaluate()

    def remap(self, indices: np.ndarray) -> np.ndarray:
        mapped = self.slot_table[indices]
        if (mapped < 0).any():
            raise RuntimeError(f"layer {self.layer}: selected expert is not resident")
        return mapped.astype(np.uint32, copy=False)


def _streaming_module(pool: ExpertPool):
    import mlx.core as mx
    import mlx.nn as nn

    class StreamingSwitchMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self._pool = pool

        def _forward(self, x, indices, host_indices):
            selected = list(dict.fromkeys(int(i) for i in host_indices.flat))
            if len(selected) > pool.slots:
                raise RuntimeError(
                    f"layer {pool.layer} needs {len(selected)} experts but has "
                    f"{pool.slots} slots; use prefill_step_size=1"
                )
            pool.ensure(selected)
            slot_indices = mx.array(pool.remap(host_indices), dtype=mx.uint32)
            x = mx.expand_dims(x, (-2, -3))

            def qmm(value, projection):
                tensors = pool.tensors[projection]
                input_global_scale = pool.input_global_scale_arrays[projection]
                qvalue, value_scales = mx.quantize(
                    value,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=input_global_scale,
                )
                value = mx.dequantize(
                    qvalue,
                    value_scales,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=input_global_scale,
                    dtype=value.dtype,
                )
                return mx.gather_qmm(
                    value,
                    tensors["weight"],
                    tensors["scales"],
                    None,
                    rhs_indices=slot_indices,
                    transpose=True,
                    group_size=16,
                    bits=4,
                    mode="nvfp4",
                    global_scale=tensors["global_scale"],
                    sorted_indices=False,
                )

            return qmm(nn.relu2(qmm(x, "up_proj")), "down_proj").squeeze(-2)

        def __call__(self, x, indices):
            host = np.asarray(indices)
            selected = np.unique(host).size
            if selected <= pool.slots:
                return self._forward(x, indices, host)
            if host.ndim != 3 or host.shape[0] != 1:
                raise RuntimeError("oversized expert set only supports batch size 1")
            pieces = []
            for i in range(host.shape[1]):
                piece = self._forward(
                    x[:, i : i + 1],
                    indices[:, i : i + 1],
                    host[:, i : i + 1],
                )
                # Materialize before a later token can replace its expert slots.
                mx.eval(piece)
                pieces.append(piece)
            return mx.concatenate(pieces, axis=1)

    return StreamingSwitchMLP()


def _static_nvfp4_module(weight_shape, scale_shape, bias: bool):
    """One resident packed NVFP4 linear with calibrated W4A4 inputs."""
    import mlx.core as mx
    import mlx.nn as nn

    class StaticNVFP4Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((1,) + tuple(weight_shape), dtype=mx.uint32)
            self.scales = mx.zeros((1,) + tuple(scale_shape), dtype=mx.uint8)
            self.global_scale = mx.ones((1,), dtype=mx.float32)
            self.input_global_scale = mx.ones((), dtype=mx.float32)
            if bias:
                self.bias = mx.zeros((weight_shape[0],))

        def __call__(self, x):
            qx, x_scales = mx.quantize(
                x,
                group_size=16,
                bits=4,
                mode="nvfp4",
                global_scale=self.input_global_scale,
            )
            x = mx.dequantize(
                qx,
                x_scales,
                group_size=16,
                bits=4,
                mode="nvfp4",
                global_scale=self.input_global_scale,
                dtype=x.dtype,
            )
            indices = mx.zeros(x.shape[:-1] + (1,), dtype=mx.uint32)
            x = mx.expand_dims(x, (-2, -3))
            x = mx.gather_qmm(
                x,
                self.weight,
                self.scales,
                None,
                rhs_indices=indices,
                transpose=True,
                group_size=16,
                bits=4,
                mode="nvfp4",
                global_scale=self.global_scale,
                sorted_indices=False,
            ).squeeze(-2).squeeze(-2)
            if "bias" in self:
                x = x + self.bias
            return x

    return StaticNVFP4Linear()


def _resident_sanitizer(pools: dict[int, ExpertPool]):
    import mlx.core as mx

    def sanitize(model, weights):
        for layer, pool in pools.items():
            model.backbone.layers[layer].mixer.switch_mlp = _streaming_module(pool)

        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("mtp.") and ".mixer.experts." not in key
        }

        # The authoritative Nemotron implementation routes in FP32. Keeping a
        # BF16 gate matmul can change the 22 selected experts out of 512.
        for key in weights:
            if key.endswith(".mixer.gate.weight"):
                weights[key] = weights[key].astype(mx.float32)

        # This checkpoint has one non-routed NVFP4 tensor. Keep it packed and
        # preserve its calibrated activation scale just like the routed bank.
        static_prefix = "backbone.layers.1.mixer.shared_experts.down_proj"
        static_scale2 = f"{static_prefix}.weight_scale_2"
        if static_scale2 in weights:
            weight_key = f"{static_prefix}.weight"
            scale_key = f"{static_prefix}.weight_scale"
            input_key = f"{static_prefix}.input_scale"
            weight = weights.pop(weight_key).view(mx.uint32)
            scales = weights.pop(scale_key).view(mx.uint8)
            model.backbone.layers[1].mixer.shared_experts.down_proj = (
                _static_nvfp4_module(
                    weight.shape,
                    scales.shape,
                    f"{static_prefix}.bias" in weights,
                )
            )
            weights[weight_key] = mx.expand_dims(weight, 0)
            weights[f"{static_prefix}.scales"] = mx.expand_dims(scales, 0)
            weights[f"{static_prefix}.global_scale"] = (
                weights.pop(static_scale2).astype(mx.float32).reshape((1,)) * 2688.0
            )
            weights[f"{static_prefix}.input_global_scale"] = (
                weights.pop(input_key).astype(mx.float32).reshape(()) * 2688.0
            )

        for key in list(weights):
            if not key.endswith(".weight"):
                continue
            prefix = key[: -len(".weight")]
            scale_key = f"{prefix}.weight_scale"
            scale2_key = f"{prefix}.weight_scale_2"
            if scale2_key in weights:
                raise ValueError(
                    f"unsupported non-routed NVFP4 tensor {prefix}; refusing to "
                    "discard its input_scale"
                )
            elif scale_key in weights:
                weights[key] = (
                    mx.from_fp8(weights[key], dtype=mx.float32)
                    * weights[scale_key].astype(mx.float32)
                ).astype(mx.bfloat16)

        weights = {
            key: value
            for key, value in weights.items()
            if not key.endswith(
                (".input_scale", ".weight_scale", ".weight_scale_2", ".k_scale", ".v_scale")
            )
        }
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
        return weights

    return sanitize


def load_streaming_model(
    model_dir: str | Path,
    *,
    expert_budget_gib: float = 8.0,
    workers: int = 6,
    trust_remote_code: bool = False,
):
    """Load resident weights and install direct-from-source expert pools."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import nemotron_h

    if "global_scale" not in (mx.gather_qmm.__doc__ or ""):
        raise RuntimeError(
            "this runtime requires MLX PR #4458: gather_qmm has no "
            "per-expert global_scale argument"
        )

    model_dir = Path(model_dir).resolve()
    index = ModelOptIndex(model_dir)
    summary = index.validate_experts()
    budget = int(expert_budget_gib * (1 << 30))
    slots = min(index.num_experts, budget // summary["bytes_per_slot_set"])
    if slots < index.top_k:
        minimum = index.top_k * summary["bytes_per_slot_set"] / (1 << 30)
        raise ValueError(
            f"expert budget gives {slots} slots/layer; top-k needs {index.top_k}. "
            f"Use at least {minimum:.2f} GiB"
        )

    reader = ExpertReader(index, workers=workers)
    try:
        pools = {
            layer: ExpertPool(index, reader, layer, slots)
            for layer in index.moe_layers
        }
        with _SANITIZER_LOCK:
            previous = nemotron_h.Model.sanitize
            nemotron_h.Model.sanitize = _resident_sanitizer(pools)
            try:
                model, tokenizer = load(
                    str(model_dir), lazy=True, trust_remote_code=trust_remote_code
                )
            finally:
                nemotron_h.Model.sanitize = previous

        mx.eval(model.parameters())
        for pool in pools.values():
            pool.evaluate()
        return model, tokenizer, pools, reader
    except BaseException:
        reader.close()
        raise


@contextlib.contextmanager
def streaming_model(*args, **kwargs):
    model, tokenizer, pools, reader = load_streaming_model(*args, **kwargs)
    try:
        yield model, tokenizer, pools, reader
    finally:
        reader.close()


def aggregate_stats(pools: dict[int, ExpertPool], reader: ExpertReader) -> dict:
    return {
        "hits": sum(pool.hits for pool in pools.values()),
        "misses": sum(pool.misses for pool in pools.values()),
        "evictions": sum(pool.evictions for pool in pools.values()),
        "bytes_read": reader.bytes_read,
        "useful_bytes": reader.useful_bytes,
        "reads": reader.reads,
    }
