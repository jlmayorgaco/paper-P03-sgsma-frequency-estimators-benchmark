from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines import benchmark_definition as bd


def test_pi_gru_missing_torch_raises_clear_runtime_error(monkeypatch) -> None:
    pi_gru_spec = next(spec for spec in bd.ACTIVE_ESTIMATOR_SPECS if spec.key == "pi_gru")
    monkeypatch.setattr(bd, "ACTIVE_ESTIMATOR_SPECS", (pi_gru_spec,))

    def fake_import_module(name: str, package=None):  # noqa: ANN001
        if name == "estimators.pi_gru":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return importlib.import_module(name, package)

    monkeypatch.setattr(bd.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match="PI-GRU.*requires torch"):
        bd.load_active_estimators()
