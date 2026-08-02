from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_sampling import TARGET_SPECIFIC_METHODS, run_sampling
from source import PAPER_METHODS


def main() -> None:
    for method_index, method_name in enumerate(PAPER_METHODS):
        semivalues = (
            ["BT-1-1"]
            if method_name in TARGET_SPECIFIC_METHODS
            else ["BT-1-1", "WBZ-05"]
        )
        try:
            result = run_sampling(
                {
                    "dataset": "soug",
                    "method": method_name,
                    "semivalues": semivalues,
                    "uep": 100,
                    "seed": 20260702 + method_index,
                    "size_distribution": "ofaa",
                    "boundary_mode": "exact",
                }
            )
            estimates = result["estimates"]
            if result["status"] != "ok" or result["sampling_passes"] != 1:
                print("fail")
                raise SystemExit(1)
            for estimate in estimates.values():
                if estimate.shape != (50,) or not np.all(np.isfinite(estimate)):
                    print("fail")
                    raise SystemExit(1)
        except Exception:
            print("fail")
            raise SystemExit(1)

    print("pass")


if __name__ == "__main__":
    main()
