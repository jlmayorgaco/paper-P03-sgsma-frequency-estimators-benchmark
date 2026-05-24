from __future__ import annotations


class MyEstimator:
    name = "MyEstimator"

    def __init__(self, f0_hz: float = 60.0) -> None:
        self.f0_hz = float(f0_hz)

    @classmethod
    def default_params(cls) -> dict[str, float]:
        return {"f0_hz": 60.0}

    def reset(self) -> None:
        return None

    def step(self, z: float, t_s: float | None = None, memory: object | None = None) -> float:
        return self.f0_hz

