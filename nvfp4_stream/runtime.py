# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt static weights and manage routed experts for MLX inference."""

from __future__ import annotations

import contextlib
import os
import threading
from collections import OrderedDict

import numpy as np

from .index import NVFP4_SCALE_DENOM, PROJECTIONS, ModelOptIndex
from .reader import ExpertReader


_SANITIZER_LOCK = threading.Lock()
_RESIDENT_RESERVE_BYTES = 4 << 30


def select_expert_mode(
    requested: str, source_bytes: int, memory_bytes: int | None = None
) -> str:
    """Resolve an explicit or automatic routed-expert residency mode."""
    if requested not in ("auto", "resident", "stream"):
        raise ValueError(f"invalid expert mode: {requested}")
    if requested != "auto":
        return requested
    if memory_bytes is None:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    resident_cap = memory_bytes * 3 // 4 - _RESIDENT_RESERVE_BYTES
    return "resident" if source_bytes <= resident_cap else "stream"


class ExpertPool:
    """Cache one MoE layer's experts in fixed slots."""

    def __init__(
        self,
        index: ModelOptIndex,
        reader: ExpertReader,
        layer: int,
        slots: int,
        expert_stats: bool = False,
    ):
        import mlx.core as mx

        self.index = index
        self.reader = reader
        self.layer = layer
        self.formats = index.expert_formats[layer]
        self.slots = slots
        self.id_to_slot: OrderedDict[int, int] = OrderedDict()
        self.free = list(range(slots - 1, -1, -1))
        self.slot_table = np.full(index.num_experts, -1, dtype=np.int32)
        self.fully_resident = False
        self.expert_stats = expert_stats
        self.hits = self.misses = self.evictions = 0
        self.cold_misses = self.capacity_misses = 0
        self._run_accessed_experts: set[int] = set()
        self._seen_experts: set[int] = set()
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

    def reset_stats(self) -> None:
        """Reset per-run counters without changing cache contents or LRU state."""
        self.hits = self.misses = self.evictions = 0
        self.cold_misses = self.capacity_misses = 0
        self._run_accessed_experts.clear()

    def evaluate(self) -> None:
        """Materialize all MLX slot tensors and NVFP4 input-scale scalars."""
        import mlx.core as mx

        tensors = [
            tensor for parts in self.tensors.values() for tensor in parts.values()
        ]
        tensors.extend(self.input_global_scale_arrays.values())
        mx.eval(tensors)

    def _record_input_scale(self, projection: str, value: np.float32) -> None:
        """Record one shared NVFP4 activation scale for a projection."""
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
        """Assign an expert to a free or oldest unprotected LRU slot."""
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
        """Ensure requested experts occupy materialized cache slots."""
        if self.expert_stats:
            self._run_accessed_experts.update(experts)
        wanted = set(experts)
        missing = []
        for expert in experts:
            if expert in self.id_to_slot:
                self.id_to_slot.move_to_end(expert)
                self.hits += 1
            else:
                self.misses += 1
                if self.expert_stats:
                    if expert in self._seen_experts:
                        self.capacity_misses += 1
                    else:
                        self.cold_misses += 1
                        self._seen_experts.add(expert)
                self._allocate(expert, wanted)
                missing.append(expert)
        if not missing:
            return

        rows = self.reader.read_experts(self.layer, missing)
        for expert in missing:
            slot = self.id_to_slot[expert]
            for projection in PROJECTIONS:
                row = rows[expert][projection]
                self._record_input_scale(
                    projection,
                    row["input_global_scale"],
                )
                for part in self.tensors[projection]:
                    self.tensors[projection][part][slot] = row[part]
        self.evaluate()

    def preload_all(self) -> None:
        """Load and materialize a complete identity-indexed expert bank."""
        import mlx.core as mx

        if self.slots != self.index.num_experts:
            raise ValueError("resident expert pool needs one slot per expert")
        experts = list(range(self.index.num_experts))
        rows = self.reader.read_experts(self.layer, experts)
        for projection in PROJECTIONS:
            for expert in experts:
                self._record_input_scale(
                    projection, rows[expert][projection]["input_global_scale"]
                )
            for part in self.tensors[projection]:
                self.tensors[projection][part] = mx.stack(
                    [rows[expert][projection][part] for expert in experts]
                )
        self.id_to_slot = OrderedDict((expert, expert) for expert in experts)
        self.free.clear()
        self.slot_table = np.arange(self.index.num_experts, dtype=np.int32)
        self.fully_resident = True
        self.evaluate()

    def remap(self, indices: np.ndarray) -> np.ndarray:
        """Map host router expert IDs to current unsigned cache-slot IDs."""
        mapped = self.slot_table[indices]
        if (mapped < 0).any():
            raise RuntimeError(f"layer {self.layer}: selected expert is not resident")
        return mapped.astype(np.uint32, copy=False)


def _nvfp4_qmm(value, tensors, input_global_scale, indices):
    """Apply NVFP4 activation rounding and selected packed-weight projection."""
    import mlx.core as mx

    output_dtype = value.dtype
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
    kwargs = (
        {"global_scale": tensors["global_scale"]}
        if mx.default_device() == mx.gpu
        else {}
    )
    value = mx.gather_qmm(
        value,
        tensors["weight"],
        tensors["scales"],
        None,
        rhs_indices=indices,
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
        sorted_indices=False,
        **kwargs,
    )
    if not kwargs:
        scale = tensors["global_scale"][indices] / NVFP4_SCALE_DENOM
        value = (value * mx.expand_dims(scale, (-1, -2))).astype(output_dtype)
    return value


def _expert_module(pool: ExpertPool):
    """Build a routed MLP that reads weights from one expert pool."""
    import mlx.core as mx
    import mlx.nn as nn

    class ExpertSwitchMLP(nn.Module):
        """Run one MoE routed MLP from an expert pool."""
        def __init__(self):
            super().__init__()
            self._pool = pool

        def _forward(self, x, slot_indices):
            """Compute down-projection of ReLU2 up-projection by slot ID."""
            x = mx.expand_dims(x, (-2, -3))

            def qmm(value, projection):
                return _nvfp4_qmm(
                    value,
                    pool.tensors[projection],
                    pool.input_global_scale_arrays[projection],
                    slot_indices,
                )

            return qmm(nn.relu2(qmm(x, "up_proj")), "down_proj").squeeze(-2)

        def __call__(self, x, indices):
            """Execute direct resident slots or populate and remap a stream cache."""
            if pool.fully_resident:
                return self._forward(x, indices.astype(mx.uint32))
            host = np.asarray(indices)
            selected = np.unique(host).size
            if selected <= pool.slots:
                wanted = list(dict.fromkeys(int(i) for i in host.flat))
                pool.ensure(wanted)
                mapped = mx.array(pool.remap(host), dtype=mx.uint32)
                return self._forward(x, mapped)
            if host.ndim != 3 or host.shape[0] != 1:
                raise RuntimeError("oversized expert set only supports batch size 1")
            pieces = []
            for i in range(host.shape[1]):
                piece_host = host[:, i : i + 1]
                wanted = list(dict.fromkeys(int(j) for j in piece_host.flat))
                pool.ensure(wanted)
                mapped = mx.array(pool.remap(piece_host), dtype=mx.uint32)
                piece = self._forward(x[:, i : i + 1], mapped)
                # Materialize before a later token can replace its expert slots.
                mx.eval(piece)
                pieces.append(piece)
            return mx.concatenate(pieces, axis=1)

    return ExpertSwitchMLP()


def _static_nvfp4_module(weight_shape, scale_shape, bias: bool):
    """Build a one-entry resident NVFP4 linear for a non-routed weight."""
    import mlx.core as mx
    import mlx.nn as nn

    class StaticNVFP4Linear(nn.Module):
        """Represent one static NVFP4 projection as a one-entry bank."""
        def __init__(self):
            super().__init__()
            self.weight = mx.zeros((1,) + tuple(weight_shape), dtype=mx.uint32)
            self.scales = mx.zeros((1,) + tuple(scale_shape), dtype=mx.uint8)
            self.global_scale = mx.ones((1,), dtype=mx.float32)
            self.input_global_scale = mx.ones((), dtype=mx.float32)
            if bias:
                self.bias = mx.zeros((weight_shape[0],))

        def __call__(self, x):
            """Apply the sole NVFP4 bank entry at every input position."""
            indices = mx.zeros(x.shape[:-1] + (1,), dtype=mx.uint32)
            x = mx.expand_dims(x, (-2, -3))
            x = _nvfp4_qmm(
                x,
                {
                    "weight": self.weight,
                    "scales": self.scales,
                    "global_scale": self.global_scale,
                },
                self.input_global_scale,
                indices,
            ).squeeze(-2).squeeze(-2)
            if "bias" in self:
                x = x + self.bias
            return x

    return StaticNVFP4Linear()


def _resident_sanitizer(pools: dict[int, ExpertPool], base_sanitize=None):
    """Build the temporary in-memory sanitizer for dense and MoE loading."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    def sanitize(model, weights):
        """Adapt lazy in-memory weights and modules to the runtime layout."""
        if base_sanitize is not None:
            weights = base_sanitize(model, weights)
        for layer, pool in pools.items():
            model.backbone.layers[layer].mixer.switch_mlp = _expert_module(pool)

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

        replacements = []
        for static_scale2 in [
            key for key in weights if key.endswith(".weight_scale_2")
        ]:
            static_prefix = static_scale2[: -len(".weight_scale_2")]
            weight_key = f"{static_prefix}.weight"
            scale_key = f"{static_prefix}.weight_scale"
            input_key = f"{static_prefix}.input_scale"
            weight = weights.pop(weight_key).view(mx.uint32)
            scales = weights.pop(scale_key).view(mx.uint8)
            replacements.append(
                (
                    static_prefix,
                    _static_nvfp4_module(
                        weight.shape,
                        scales.shape,
                        f"{static_prefix}.bias" in weights,
                    ),
                )
            )
            weights[weight_key] = mx.expand_dims(weight, 0)
            weights[f"{static_prefix}.scales"] = mx.expand_dims(scales, 0)
            weights[f"{static_prefix}.global_scale"] = (
                weights.pop(static_scale2).astype(mx.float32).reshape((1,))
                * NVFP4_SCALE_DENOM
            )
            weights[f"{static_prefix}.input_global_scale"] = (
                weights.pop(input_key).astype(mx.float32).reshape(())
                * NVFP4_SCALE_DENOM
            )
        if replacements:
            model.update_modules(tree_unflatten(replacements))

        for key in list(weights):
            if not key.endswith(".weight"):
                continue
            prefix = key[: -len(".weight")]
            scale_key = f"{prefix}.weight_scale"
            if scale_key in weights:
                weights[key] = (
                    mx.from_fp8(weights[key], dtype=mx.float32)
                    * weights[scale_key].astype(mx.float32)
                ).astype(mx.bfloat16)

        weights = {
            key: value
            for key, value in weights.items()
            if not key.endswith(
                (
                    ".input_scale",
                    ".weight_scale",
                    ".weight_scale_2",
                    ".k_scale",
                    ".v_scale",
                )
            )
        }
        for key, value in list(weights.items()):
            if "conv1d.weight" in key and value.shape[-1] != 1:
                weights[key] = value.moveaxis(2, 1)
        return weights

    return sanitize


def load_streaming_model(
    index: ModelOptIndex,
    *,
    expert_budget_gib: float = 8.0,
    expert_mode: str = "auto",
    workers: int = 6,
    trust_remote_code: bool = False,
    expert_stats: bool = False,
):
    """Load common weights and prepare dense or pool-backed MoE inference."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import llama, nemotron_h

    if mx.default_device() == mx.gpu and "global_scale" not in (
        mx.gather_qmm.__doc__ or ""
    ):
        raise RuntimeError(
            "this runtime requires MLX PR #4458: gather_qmm has no "
            "per-expert global_scale argument"
        )

    model_dir = index.model_dir
    model_module = llama if index.model_type == "llama" else nemotron_h
    summary = index.expert_summary
    expert_mode = select_expert_mode(
        expert_mode if index.moe_layers else "resident", index.source_bytes
    )
    budget = int(expert_budget_gib * (1 << 30))
    slots = (
        index.num_experts
        if expert_mode == "resident"
        else min(index.num_experts, budget // summary["bytes_per_slot_set"])
    )
    if expert_mode == "stream" and slots < index.top_k:
        minimum = index.top_k * summary["bytes_per_slot_set"] / (1 << 30)
        raise ValueError(
            f"expert budget gives {slots} slots/layer; top-k needs {index.top_k}. "
            f"Use at least {minimum:.2f} GiB"
        )

    reader = ExpertReader(index, workers=workers)
    try:
        pools = {
            layer: ExpertPool(
                index, reader, layer, slots, expert_stats=expert_stats
            )
            for layer in index.moe_layers
        }
        with _SANITIZER_LOCK:
            previous = model_module.Model.sanitize
            model_module.Model.sanitize = _resident_sanitizer(
                pools, previous if index.model_type == "llama" else None
            )
            try:
                model, tokenizer = load(
                    str(model_dir),
                    lazy=True,
                    trust_remote_code=trust_remote_code,
                    model_config={"num_hidden_layers": index.num_layers},
                )
            finally:
                model_module.Model.sanitize = previous

        mx.eval(model.parameters())
        if expert_mode == "resident":
            for pool in pools.values():
                pool.preload_all()
            reader.close()
        else:
            for pool in pools.values():
                pool.evaluate()
        return model, tokenizer, pools, reader
    except BaseException:
        reader.close()
        raise


@contextlib.contextmanager
def streaming_model(*args, **kwargs):
    """Yield loaded inference resources and always close their expert reader."""
    model, tokenizer, pools, reader = load_streaming_model(*args, **kwargs)
    try:
        yield model, tokenizer, pools, reader
    finally:
        reader.close()


def aggregate_stats(
    pools: dict[int, ExpertPool], reader: ExpertReader, expert_stats: bool = False
) -> dict:
    """Aggregate cache and reader counters across every MoE layer pool."""
    expert_mode = (
        "resident" if all(pool.fully_resident for pool in pools.values()) else "stream"
    )
    stats = {
        "expert_mode": expert_mode,
        "hits": sum(pool.hits for pool in pools.values()),
        "misses": sum(pool.misses for pool in pools.values()),
        "evictions": sum(pool.evictions for pool in pools.values()),
        "bytes_read": reader.bytes_read,
        "useful_bytes": reader.useful_bytes,
        "reads": reader.reads,
    }
    if expert_stats:
        stats.update(
            experts_loaded=sum(len(pool.id_to_slot) for pool in pools.values()),
            cold_misses=sum(pool.cold_misses for pool in pools.values()),
            capacity_misses=sum(pool.capacity_misses for pool in pools.values()),
        )
        if expert_mode == "stream":
            stats["unique_experts_accessed"] = sum(
                len(pool._run_accessed_experts) for pool in pools.values()
            )
    return stats
