# SPDX-License-Identifier: MIT

# Copyright (c) 2026 Oscar Montiel Ross

"""
Reproducibility script for the ZOD-derived Mediative Fuzzy Logic (MFL)
illustrative case study.

This script accompanies the manuscript:

```
Contradiction-Aware Mediative Fuzzy Logic: Operators, Semantic Coherence,
Type-2/Type-3 Extensions, and Quantum Semantics
```

Author:
Oscar Montiel Ross

License:
The code in this script is released under the MIT License.
See the LICENSE file in this repository for the full license text.

Data notice:
This script uses a minimal preprocessed representation of selected
ZOD-derived rows only to reproduce the illustrative calculations reported
in the manuscript.

```
No raw ZOD sensor data, images, LiDAR, radar, video, or personally
identifying information are included in this repository.

The original Zenseact Open Dataset (ZOD) remains governed by its own
license terms. Users should consult and cite the original ZOD dataset when
using or referencing ZOD-derived material.
```

Purpose:
Starting from zod_selected_preprocessed_rows.csv, this script recomputes:

```
    - the contextual uncertainty score c;
    - the distance and proximity quantities D and r;
    - the mediative truth-falsity pair (mu_p, nu_p);
    - hesitation pi_p;
    - contradiction zeta_p;
    - the mediative score M(mu_p, nu_p);
    - and the final safety-first decision category.

The computation is illustrative and reproducibility-oriented. It does not
train, validate, or benchmark an autonomous-driving perception model.
```

"""


from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional

INPUT_FILE = Path("zod_selected_preprocessed_rows.csv")
OUTPUT_FILE = Path("zod_mfl_recomputed_results.csv")


def clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def parse_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if value == "":
        return None
    return float(value)


def compute_context_score(row: Dict[str, str]) -> float:
    """Compute c = 0.35 c_weather + 0.25 c_light + 0.20 c_road + 0.20 c_density."""
    c_weather = float(row["c_weather"])
    c_light = float(row["c_light"])
    c_road = float(row["c_road"])
    c_density = float(row["c_density"])
    return 0.35 * c_weather + 0.25 * c_light + 0.20 * c_road + 0.20 * c_density


def compute_truth_falsity(row: Dict[str, str], c: float) -> Dict[str, Optional[float]]:
    risk_indicator = int(row["risk_indicator"])
    distance_m = parse_float(row.get("distance_m", ""))

    if risk_indicator == 1:
        if distance_m is None:
            raise ValueError("Positive-risk rows require distance_m.")
        D = clip(distance_m / 10.0, 0.0, 1.0)
        r = 1.0 - D
        mu_p = 0.55 + 0.35 * r + 0.10 * c
        nu_p = 0.05 + 0.20 * D + 0.25 * c
    else:
        D = None
        r = None
        mu_p = 0.10 + 0.30 * c
        nu_p = 0.60 + 0.25 * (1.0 - c)

    pi_p = max(0.0, 1.0 - mu_p - nu_p)
    zeta_p = max(0.0, mu_p + nu_p - 1.0)
    M = (1.0 - pi_p - zeta_p / 2.0) * mu_p + (pi_p + zeta_p / 2.0) * (1.0 - nu_p)

    return {
        "D": D,
        "r": r,
        "mu_p": mu_p,
        "nu_p": nu_p,
        "pi_p": pi_p,
        "zeta_p": zeta_p,
        "M": M,
    }


def decision_from_score(M: float) -> str:
    if M >= 0.7:
        return "decisive braking"
    if M >= 0.5:
        return "cautious deceleration and additional sensing"
    return "cautious monitoring"


def round3(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Cannot find {INPUT_FILE}. Run this script in the supplement folder.")

    rows_out = []
    with INPUT_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = compute_context_score(row)
            computed = compute_truth_falsity(row, c)
            decision = decision_from_score(computed["M"])

            rows_out.append({
                "case_id": row["case_id"],
                "split": row["split"],
                "frame_id": row["frame_id"],
                "risk_indicator": row["risk_indicator"],
                "reason_field": row["reason_field"],
                "weather": row["weather"],
                "time_of_day": row["time_of_day"],
                "road_condition": row["road_condition"],
                "context_score_c": round3(c),
                "D": round3(computed["D"]),
                "r": round3(computed["r"]),
                "mu_p": round3(computed["mu_p"]),
                "nu_p": round3(computed["nu_p"]),
                "pi_p": round3(computed["pi_p"]),
                "zeta_p": round3(computed["zeta_p"]),
                "M": round3(computed["M"]),
                "decision": decision,
            })

    fieldnames = list(rows_out[0].keys())
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {OUTPUT_FILE}")
    for row in rows_out:
        print(
            f"Case {row['case_id']}: c={row['context_score_c']}, "
            f"(mu_p,nu_p)=({row['mu_p']},{row['nu_p']}), "
            f"pi={row['pi_p']}, zeta={row['zeta_p']}, M={row['M']}, "
            f"decision={row['decision']}"
        )


if __name__ == "__main__":
    main()
