# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Read routed expert ranges directly with parallel, coalesced positional I/O."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .index import NVFP4_SCALE_DENOM, PROJECTIONS, ModelOptIndex, TensorLoc


class ExpertReader:
    """Read selected routed-expert ranges from original safetensors shards."""
    def __init__(
        self,
        index: ModelOptIndex,
        workers: int = 6,
        merge_gap: int = 64 * 1024,
        max_span: int = 16 * 1024 * 1024,
    ):
        self.index = index
        self.merge_gap = merge_gap
        self.max_span = max_span
        self._fds: dict[str, int] = {}
        self._fd_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self.bytes_read = 0
        self.useful_bytes = 0
        self.reads = 0

    def reset_stats(self) -> None:
        """Clear accumulated physical and useful-byte read counters."""
        self.bytes_read = self.useful_bytes = self.reads = 0

    def _fd(self, path: str) -> int:
        """Lazily open or reuse a read-only descriptor for one shard."""
        with self._fd_lock:
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            return self._fds[path]

    def _plan(self, requests):
        """Merge nearby same-shard requests into bounded physical read spans."""
        spans = []
        for key, path, start, size in sorted(requests, key=lambda item: item[1:3]):
            if spans:
                old_path, old_start, old_size, members = spans[-1]
                gap = start - (old_start + old_size)
                merged_size = start + size - old_start
                if (
                    path == old_path
                    and 0 <= gap <= self.merge_gap
                    and merged_size <= self.max_span
                ):
                    spans[-1] = (
                        old_path,
                        old_start,
                        merged_size,
                        members + [(key, start - old_start, size)],
                    )
                    continue
            spans.append((path, start, size, [(key, 0, size)]))
        return spans

    def _read_ranges(self, requests) -> dict:
        """Execute planned spans and split their bytes into requested views."""
        def read_span(span):
            """Read one planned shard span with positional I/O."""
            path, start, size, members = span
            data = os.pread(self._fd(path), size, start)
            if len(data) != size:
                raise IOError(f"short read {len(data)}/{size} at {start} in {path}")
            return data, members

        output = {}
        futures = [self._executor.submit(read_span, span) for span in self._plan(requests)]
        for future in futures:
            data, members = future.result()
            view = memoryview(data)
            self.bytes_read += len(data)
            self.reads += 1
            for key, offset, size in members:
                output[key] = view[offset : offset + size]
                self.useful_bytes += size
        return output

    @staticmethod
    def _weight_np(raw: memoryview, loc: TensorLoc) -> np.ndarray:
        """Expose a routed weight as a NumPy view of its exact stored bits."""
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(loc.shape)
        return u8.view("<u4").reshape(loc.shape[:-1] + (loc.shape[-1] // 4,))

    @staticmethod
    def _scales_np(raw: memoryview, loc: TensorLoc) -> np.ndarray:
        """Expose an NVFP4 block-scale payload as an unchanged U8 view."""
        return np.frombuffer(raw, dtype=np.uint8).reshape(loc.shape)

    @staticmethod
    def _global_scale_value(raw: memoryview, loc: TensorLoc) -> np.float32:
        """Convert a ModelOpt NVFP4 scalar scale to MLX's amax convention."""
        return np.float32(
            np.frombuffer(raw, dtype="<f4", count=1)[0] * NVFP4_SCALE_DENOM
        )

    def read_experts_numpy(self, layer: int, experts: list[int]) -> dict:
        """Synchronously load selected routed experts into host NumPy mappings."""
        requests = []
        locations = {}
        for expert in experts:
            for projection in PROJECTIONS:
                parts = self.index.expert_parts(layer, expert, projection)
                for part, loc in parts.items():
                    key = (expert, projection, part)
                    locations[key] = loc
                    requests.append((key, loc.path, loc.start, loc.nbytes))
        raw = self._read_ranges(requests)
        output = {}
        for expert in experts:
            output[expert] = {}
            for projection in PROJECTIONS:
                key = (expert, projection)
                if self.index.expert_formats[layer][projection] == "bf16":
                    weight = locations[key + ("weight",)]
                    output[expert][projection] = {
                        "weight": np.frombuffer(
                            raw[key + ("weight",)], dtype="<u2"
                        ).reshape(weight.shape)
                    }
                    continue
                parts = {
                    "weight": self._weight_np(
                        raw[key + ("weight",)], locations[key + ("weight",)]
                    ),
                }
                parts.update(
                    scales=self._scales_np(
                        raw[key + ("weight_scale",)],
                        locations[key + ("weight_scale",)],
                    ),
                    global_scale=self._global_scale_value(
                        raw[key + ("weight_scale_2",)],
                        locations[key + ("weight_scale_2",)],
                    ),
                    input_global_scale=self._global_scale_value(
                        raw[key + ("input_scale",)],
                        locations[key + ("input_scale",)],
                    ),
                )
                output[expert][projection] = parts
        return output

    def read_experts(self, layer: int, experts: list[int]) -> dict:
        """Load selected routed experts into their MLX runtime representation."""
        import mlx.core as mx

        arrays = self.read_experts_numpy(layer, experts)
        output = {}
        for expert, projections in arrays.items():
            output[expert] = {}
            for projection, parts in projections.items():
                weight = mx.array(parts["weight"])
                if self.index.expert_formats[layer][projection] == "bf16":
                    output[expert][projection] = {"weight": weight.view(mx.bfloat16)}
                    continue
                row = {"weight": weight}
                row.update(
                    scales=mx.array(parts["scales"]),
                    global_scale=mx.array(parts["global_scale"], mx.float32),
                    input_global_scale=parts["input_global_scale"],
                )
                output[expert][projection] = row
        return output

    def close(self) -> None:
        """Wait for range reads, close shard descriptors, and stop workers."""
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
