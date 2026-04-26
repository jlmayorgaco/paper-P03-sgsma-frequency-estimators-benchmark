from __future__ import annotations

from abc import ABC
import sys
import time
from typing import Any


class MemoryStore:
    """
    Mutable memory object shared with estimators during step-wise execution.

    The benchmark uses it to track memory usage proxies consistently across
    all estimators (peak/current/mean bytes).
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._samples = 0
        self._sum_bytes = 0.0
        self._peak_bytes = 0
        self._last_bytes = 0

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self._data.setdefault(key, default)

    def pop(self, key: str, default: Any = None) -> Any:
        if default is None:
            return self._data.pop(key)
        return self._data.pop(key, default)

    def clear(self) -> None:
        self._data.clear()
        self._samples = 0
        self._sum_bytes = 0.0
        self._peak_bytes = 0
        self._last_bytes = 0

    def _estimate_bytes(self) -> int:
        """
        Conservative recursive size estimate over dict/list/tuple/set contents.
        """
        seen: set[int] = set()

        def _size(obj: Any) -> int:
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            total = sys.getsizeof(obj)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    total += _size(k)
                    total += _size(v)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    total += _size(item)
            return total

        return int(_size(self._data))

    def observe(self) -> None:
        current = self._estimate_bytes()
        self._last_bytes = current
        self._peak_bytes = max(self._peak_bytes, current)
        self._sum_bytes += float(current)
        self._samples += 1

    def summary(self) -> dict[str, float | int]:
        mean_bytes = (self._sum_bytes / self._samples) if self._samples > 0 else 0.0
        return {
            "samples": int(self._samples),
            "current_bytes": int(self._last_bytes),
            "peak_bytes": int(self._peak_bytes),
            "mean_bytes": float(mean_bytes),
            "key_count": int(len(self._data)),
        }


class BaseFrequencyEstimator(ABC):
    """
    Minimal common contract for all benchmark estimators.

    Every estimator should:
    - expose a human-readable name
    - implement _step(...) core logic
    - use the standardized public step(...) wrapper for fair comparison
    - report its structural latency in samples
    """

    name: str = "BaseEstimator"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Backward-compatible migration:
        if a subclass defines `step` but not `_step`, we remap it automatically.
        """
        super().__init_subclass__(**kwargs)
        if cls is BaseFrequencyEstimator:
            return

        user_step = cls.__dict__.get("step")
        user_core = cls.__dict__.get("_step")
        if user_step is not None and user_step is not BaseFrequencyEstimator.step and user_core is None:
            setattr(cls, "_step", user_step)
            setattr(cls, "step", BaseFrequencyEstimator.step)

    def _ensure_runtime_state(self) -> None:
        if not hasattr(self, "_rt_initialized"):
            self._rt_initialized = True
            self._rt_step_count = 0
            self._rt_invalid_count = 0
            self._rt_first_valid_step = None
            self._rt_step_times_us = []

    def _invoke_core_step(self, z: float, t_s: float | None, memory: MemoryStore) -> float:
        core = getattr(self, "_step", None)
        if core is None:
            raise NotImplementedError(f"{self.__class__.__name__} must implement _step(...)")

        try:
            return float(core(z, t_s, memory))
        except TypeError:
            try:
                return float(core(z, t_s))
            except TypeError:
                return float(core(z))

    def step(self, z: float, t_s: float | None = None, memory: MemoryStore | None = None) -> float:
        """
        Process one input sample through the standardized anti-cheat wrapper.
        This is the only public stepping API used by the benchmark.
        """
        self._ensure_runtime_state()
        mem = memory if memory is not None else MemoryStore()

        t0 = time.process_time()
        y = self._invoke_core_step(float(z), t_s, mem)
        elapsed_us = (time.process_time() - t0) * 1e6

        self._rt_step_count += 1
        self._rt_step_times_us.append(float(elapsed_us))
        if not (y == y and abs(y) != float("inf")):
            self._rt_invalid_count += 1
        elif self._rt_first_valid_step is None:
            self._rt_first_valid_step = self._rt_step_count

        mem.observe()
        return float(y)

    def runtime_summary(self, memory: MemoryStore | None = None) -> dict[str, float | int]:
        self._ensure_runtime_state()
        arr = self._rt_step_times_us
        mean_us = float(sum(arr) / len(arr)) if arr else 0.0
        if arr and len(arr) > 1:
            var = sum((x - mean_us) ** 2 for x in arr) / len(arr)
            jitter_us = float(var ** 0.5)
        else:
            jitter_us = 0.0

        invalid_rate = (
            float(self._rt_invalid_count) / float(self._rt_step_count)
            if self._rt_step_count > 0
            else 0.0
        )
        startup_valid = int(self._rt_first_valid_step or 0)
        mem_stats = (memory.summary() if memory is not None else MemoryStore().summary())

        return {
            "runtime_steps": int(self._rt_step_count),
            "runtime_mean_step_us": mean_us,
            "runtime_jitter_us": jitter_us,
            "startup_valid_samples": startup_valid,
            "invalid_output_rate": invalid_rate,
            "memory_peak_bytes": int(mem_stats["peak_bytes"]),
            "memory_mean_bytes": float(mem_stats["mean_bytes"]),
            "memory_current_bytes": int(mem_stats["current_bytes"]),
            "memory_key_count": int(mem_stats["key_count"]),
        }

    def reset(self) -> None:
        """
        Optional state reset. Override when needed.
        """
        self._rt_initialized = False
        return None

    def structural_latency_samples(self) -> int:
        """
        Return the estimator's inherent latency in samples.
        Default = 0 for causal state-space estimators with immediate output.
        """
        return 0

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """
        Optional: default constructor parameters for config-driven instantiation.
        """
        return {}

    @classmethod
    def tuning_grid(cls) -> list[dict[str, Any]]:
        """
        Optional: estimator-specific tuning configurations.
        Default empty list; external tuning logic may override this.
        """
        return []

    @staticmethod
    def describe_params(params: dict[str, Any]) -> str:
        """
        Optional helper for serializing tuned parameters in logs/JSON.
        """
        return ",".join(f"{k}={v}" for k, v in params.items())
