from __future__ import annotations

from pipelines.benchmark_definition import build_estimator_registry_manifest


def test_experimental_estimators_are_declared() -> None:
    manifest = build_estimator_registry_manifest()
    excluded = manifest.get("excluded", [])
    labels = {row["label"] for row in excluded}
    expected = {
        "CKF",
        "SR-UKF",
        "Adaptive-EKF",
        "IMM-EKF/UKF",
        "Hinf-KF",
        "WLS-IpDFT",
        "Quinn-Fernandes",
        "Jacobsen-Interpolated-DFT",
        "Sliding-Least-Squares",
        "MUSIC",
        "Matrix-Pencil",
        "Hilbert-Phase-Derivative",
    }
    assert expected.issubset(labels)
