# Copyright (c) 2026 the nvfp4-stream authors
# SPDX-License-Identifier: Apache-2.0

"""Coalesced pread of individual expert tensors in the ModelOpt layout."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .index import NVFP4_SCALE_DENOM, PROJECTIONS, ModelOptIndex, TensorLoc


class ExpertReader:
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

    def _fd(self, path: str) -> int:
        with self._fd_lock:
            if path not in self._fds:
                self._fds[path] = os.open(path, os.O_RDONLY)
            return self._fds[path]

    def _plan(self, requests):
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
        def read_span(span):
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
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(loc.shape)
        return u8.view("<u4").reshape(loc.shape[:-1] + (loc.shape[-1] // 4,))

    @staticmethod
    def _scales_np(raw: memoryview, loc: TensorLoc) -> np.ndarray:
        return np.frombuffer(raw, dtype=np.uint8).reshape(loc.shape)

    @staticmethod
    def _global_scale_value(raw: memoryview, loc: TensorLoc) -> np.float32:
        # The ModelOpt format stores amax/(6*448). MLX's global_scale argument is amax.
        return np.float32(
            np.frombuffer(raw, dtype="<f4", count=1)[0] * NVFP4_SCALE_DENOM
        )

    def read_experts_numpy(self, layer: int, experts: list[int]) -> dict:
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
                output[expert][projection] = {
                    "weight": self._weight_np(
                        raw[key + ("weight",)], locations[key + ("weight",)]
                    ),
                    "scales": self._scales_np(
                        raw[key + ("weight_scale",)],
                        locations[key + ("weight_scale",)],
                    ),
                    "global_scale": self._global_scale_value(
                        raw[key + ("weight_scale_2",)],
                        locations[key + ("weight_scale_2",)],
                    ),
                    "input_global_scale": self._global_scale_value(
                        raw[key + ("input_scale",)],
                        locations[key + ("input_scale",)],
                    ),
                }
        return output

    def read_experts(self, layer: int, experts: list[int]) -> dict:
        import mlx.core as mx

        arrays = self.read_experts_numpy(layer, experts)
        return {
            expert: {
                projection: {
                    "weight": mx.array(parts["weight"]),
                    "scales": mx.array(parts["scales"]),
                    "global_scale": mx.array(parts["global_scale"], mx.float32),
                    "input_global_scale": parts["input_global_scale"],
                }
                for projection, parts in projections.items()
            }
            for expert, projections in arrays.items()
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
