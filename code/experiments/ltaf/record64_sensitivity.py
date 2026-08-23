"""Sensitivity of the Long-Term AF headline values to record 64.

Record 64's first rhythm annotation appears only at 18.87 h.  Seconds before the
first rhythm marker carry no rhythm evidence and are *dropped* by the
preparation script, so the estimand is time in AF divided by *analysable*
recording time and record 64 contributes only its last 5.7 h.  Its burden is
therefore an estimate on that window, not a lower bound for the whole record --
which is exactly why the record deserves a sensitivity check rather than a
caveat.

The protocols, the grid, the label weights and the value functional are taken
from :mod:`experiments.ltaf.run_ltaf` rather than reimplemented, so the shifts
reported here are shifts in the quantities the manuscript reports.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.common import save_json, environment_record  # noqa: E402
from ltaf.run_ltaf import (P_MAIN, PRIMARY_MODEL, Model, cohort_index,  # noqa: E402
                           contiguous_block, contiguous_windows, design_matrix,
                           dispersed_windows, load_af)

af, ids, _ = load_af()
ids = [str(x) for x in ids]
burden = np.array([float(x.mean()) for x in af])


def protocols_for(model: Model) -> dict[str, list]:
    """Exactly the clinical protocols of ``run_ltaf`` section (E)."""
    return {
        "1h_contiguous": contiguous_block(1.0),
        "6h_contiguous": contiguous_block(6.0),
        "4x15min": dispersed_windows(model.grid, 4, 0.25),
        "1h_contiguous_as_4x15min": contiguous_windows(4, 0.25),
        "8x15min": dispersed_windows(model.grid, 8, 0.25),
    }


def values(keep: np.ndarray) -> dict[str, float]:
    W = design_matrix(af, np.asarray(keep), P_MAIN)
    mdl = Model(W, PRIMARY_MODEL)
    out = {name: mdl.ceiling(acts) for name, acts in protocols_for(mdl).items()}
    out["equal_duration_gain_1h"] = out["4x15min"] - out["1h_contiguous"]
    out["mean_burden"] = float(W.mean())
    return out


def prefix_hours(record_id: str) -> float:
    """Unlabelled prefix of a record, read from the data summary rather than typed."""
    import csv
    path = Path(__file__).resolve().parents[2] / "results" / "data_summary.csv"
    for r in csv.DictReader(path.open()):
        if r.get("dataset") == "ltaf" and r.get("record_id") == record_id:
            return float(r["unlabelled_prefix_hours"])
    raise KeyError(f"record {record_id} not in results/data_summary.csv")


strict = cohort_index(burden, "strict")
i64 = ids.index("64")
with_64 = values(strict)
without_64 = values(np.array([i for i in strict if i != i64]))
shifts = {k: without_64[k] - with_64[k] for k in with_64}
mx = max(abs(v) for k, v in shifts.items() if k != "mean_burden")

print(f"{'quantity':>28} {'with 64':>10} {'without 64':>12} {'shift':>9}")
for k in with_64:
    print(f"{k:>28} {with_64[k]:>10.4f} {without_64[k]:>12.4f} {shifts[k]:>+9.4f}")
print(f"\nlargest shift in any protocol value: {mx:.4f}")

save_json({"environment": environment_record(),
           "estimand_note": ("prefix seconds are dropped, not coded as non-AF, so "
                             "record 64 contributes only its analysable window"),
           "record_64_burden_on_analysable_window": float(af[i64].mean()),
           "n_strict": int(strict.size),
           "n_protocols_checked": len(with_64) - 1,
           "with_record_64": with_64, "without_record_64": without_64,
           "shifts": shifts,
           "headline": {"max_abs_shift_protocol_value": float(mx),
                        "record_64_unlabelled_prefix_hours": prefix_hours("64")}},
          "ltaf_record64_sensitivity")
