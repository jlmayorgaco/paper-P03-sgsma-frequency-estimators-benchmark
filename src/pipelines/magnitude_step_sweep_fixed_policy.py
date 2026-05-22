from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipelines.voltage_step_sweep_fixed_policy import (
    configure_voltage_mag_step_fixed_policy_profile,
)


def main() -> None:
    """Canonical entry point for the final fixed-policy magnitude-step atlas."""
    configure_voltage_mag_step_fixed_policy_profile()
    from pipelines.amplitude_step_sweep import main as run_amplitude_step_sweep

    run_amplitude_step_sweep()


if __name__ == "__main__":
    main()
