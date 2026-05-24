from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol, runtime_checkable


class ContractError(TypeError):
    """Raised when public extension code violates the OpenFreqBench contract."""


@runtime_checkable
class EstimatorProtocol(Protocol):
    name: str

    def step(self, z: complex | float, t_s: float | None = None, memory: Any | None = None) -> Any:
        ...


@runtime_checkable
class ScenarioProtocol(Protocol):
    SCENARIO_NAME: str
    DEFAULT_PARAMS: dict[str, Any]

    @classmethod
    def get_name(cls) -> str:
        ...

    def run(self, seed: int | None = None) -> Any:
        ...


@dataclass(frozen=True)
class MetricProfile:
    """Public metric-profile manifest. Formulas remain implemented by platform code."""

    profile_id: str
    scope: str
    metric_ids: tuple[str, ...]
    locked: bool = True
    status: str = "active"

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


RESERVED_METRIC_PROFILES: tuple[str, ...] = (
    "three-phase-v1",
    "wams-v1",
)


def validate_estimator_contract(cls: type, *, label: str | None = None) -> None:
    name = getattr(cls, "name", label or cls.__name__)
    if not isinstance(name, str) or not name.strip():
        raise ContractError("Estimator classes must expose a non-empty string `name`.")
    if not (callable(getattr(cls, "step", None)) or callable(getattr(cls, "step_vectorized", None))):
        raise ContractError(
            f"Estimator {name!r} must define `step(...)` or `step_vectorized(...)`."
        )
    default_params = getattr(cls, "default_params", None)
    if default_params is not None and not callable(default_params):
        raise ContractError(f"Estimator {name!r} has non-callable `default_params`.")


def validate_scenario_contract(cls: type, *, label: str | None = None) -> None:
    name = label or getattr(cls, "SCENARIO_NAME", cls.__name__)
    get_name = getattr(cls, "get_name", None)
    if not callable(get_name):
        raise ContractError(f"Scenario {name!r} must define classmethod `get_name()`.")
    resolved = get_name()
    if not isinstance(resolved, str) or not resolved.strip():
        raise ContractError(f"Scenario {name!r} returned an invalid name.")
    if not callable(getattr(cls, "run", None)):
        raise ContractError(f"Scenario {resolved!r} must define `run(...)`.")
    params = getattr(cls, "DEFAULT_PARAMS", None)
    if params is not None and not isinstance(params, dict):
        raise ContractError(f"Scenario {resolved!r} DEFAULT_PARAMS must be a mapping.")


def validate_metric_profile(profile: MetricProfile, known_metric_ids: set[str]) -> None:
    if not profile.locked:
        raise ContractError(f"Metric profile {profile.profile_id!r} must be locked.")
    unknown = [metric_id for metric_id in profile.metric_ids if metric_id not in known_metric_ids]
    if unknown:
        raise ContractError(
            f"Metric profile {profile.profile_id!r} references unknown metrics: {unknown}."
        )
