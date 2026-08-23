#!/usr/bin/env python3
"""Build site/js/data.js from the frozen manuscript numbers and result files.

Every number shown on the site is read from paper/numbers.tex (the manuscript's
single source of truth) or from the released results/ directory; nothing is
typed by hand.  Re-run after `make paper` in the code package.
"""
import csv, json, re, statistics as st, collections, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "final-code-package-20260820"
NUMBERS = PKG / "manuscript/final/numbers.tex"
RES = PKG / "release/Counterfactual_Evaluation_Temporal_Protocols_Code/results"
OUT = ROOT / "site/js/data.js"

# ---------------------------------------------------------------- numbers.tex
num = {}
for m in re.finditer(r"\\newcommand\{\\(num\w+)\}\{(.*)\}", NUMBERS.read_text()):
    k, v = m.group(1), m.group(2)
    v = v.replace("\\%", "%").replace("$", "")
    num[k[3:]] = v

def f(k):
    return float(num[k].replace("+", "").replace("%", ""))

# ---------------------------------------------------------------- s3: uniform error vs m
s3 = list(csv.DictReader(open(RES / "s3_ceiling_estimation.csv")))
cal = collections.defaultdict(lambda: collections.defaultdict(list))
for r in s3:
    if r["arm"] != "known" or r["config"] != "ou_alpha0_nu0.25":
        continue
    cal[r["label"]][int(r["m"])].append(float(r["uniform_err_mean"]))
calibration = {}
for lab, d in cal.items():
    ms = sorted(d)
    calibration[lab] = {"m": ms, "eps": [st.mean(d[m]) for m in ms]}

# ---------------------------------------------------------------- s4: regret vs 2eps
s4 = list(csv.DictReader(open(RES / "s4_selection_regret.csv")))
reg = collections.defaultdict(list); env = collections.defaultdict(list)
for r in s4:
    reg[int(r["m"])].append(float(r["regret_exhaustive"]))
    env[int(r["m"])].append(float(r["bound_2eps"]))
ms = sorted(reg)
regret = {"m": ms, "regret": [st.mean(reg[m]) for m in ms],
          "envelope": [st.mean(env[m]) for m in ms]}

# ---------------------------------------------------------------- s8: nested classes
s8 = list(csv.DictReader(open(RES / "s8_resolution.csv")))
rf = collections.defaultdict(lambda: collections.defaultdict(list))
ef = collections.defaultdict(lambda: collections.defaultdict(list))
for r in s8:
    m = int(r["m"])
    for L in (1, 2, 3, 4):
        rf[L][m].append(float(r[f"regret_fixed_L{L}"]))
        ef[L][m].append(float(r[f"eps_true_L{L}"]))
ms = sorted(rf[1])
resolution = {"m": ms,
              "regret": {L: [st.mean(rf[L][m]) for m in ms] for L in (1, 2, 3, 4)},
              "eps": {L: [st.mean(ef[L][m]) for m in ms] for L in (1, 2, 3, 4)},
              "class_sizes": {1: 2, 2: 5, 3: 75, 4: 568},
              "class_names": {1: "layouts", 2: "phase", 3: "coarse bins", 4: "fine supports"}}
s8j = json.load(open(RES / "s8_resolution.json"))["headline"]
resolution["class_sizes"] = {int(k): v for k, v in s8j["class_sizes"].items()}

# ---------------------------------------------------------------- real data
cf = json.load(open(RES / "crossfit_real.json"))
sleep = {}
for lab, d in cf["sleep"].items():
    sleep[lab] = {}
    for key, v in d.items():
        meth, n = key.split("|N=")
        sleep[lab].setdefault(meth, {})[int(n)] = round(v["cross_fitted_r2"], 4)
af = {}
for key, v in cf.get("af", cf.get("ltaf", {})).items() if isinstance(cf.get("af", cf.get("ltaf")), dict) else []:
    af[key] = v
# AF headline numbers come from numbers.tex (the manuscript quotes them directly)
af_curve = {
    "fraction_pct": [1.04, 2.08, 4.17, 8.33, 16.67, 33.33],
    "hours": [0.25, 0.5, 1, 2, 4, 8],
}
# find AF series in crossfit json regardless of its key name
for k in cf:
    if k.lower().startswith(("af", "ltaf")) and isinstance(cf[k], dict):
        af_raw = cf[k]
        break
else:
    af_raw = {}

# ---------------------------------------------------------------- design table
dm = list(csv.DictReader(open(RES / "s5_design_method_summary.csv")))
design_methods = [{"method": r["method"], "min": float(r["minimum"]),
                   "mean": float(r["mean"]), "median": float(r["median"])} for r in dm]

# ---------------------------------------------------------------- identifiability instance
fig1 = json.load(open(RES / "fig1_framework.json"))

# ---------------------------------------------------------------- Sleep support-stability sweep (Figure 4)
cs = json.load(open(RES / "calibration_sweep.json"))
sweep = {"m": [], "d_kq": [], "d_kq_se": [], "d_uni": [], "d_uni_se": []}
for key in sorted(cs["sweep"], key=lambda k: cs["sweep"][k]["m_train_subjects"]):
    q = cs["sweep"][key]
    sweep["m"].append(q["m_train_subjects"])
    sweep["d_kq"].append(round(q["delta_vs_kq"], 4)); sweep["d_kq_se"].append(round(q["delta_vs_kq_se"], 4))
    sweep["d_uni"].append(round(q["delta_vs_uniform"], 4)); sweep["d_uni_se"].append(round(q["delta_vs_uniform_se"], 4))
rs = cs["repeated_subsampling"]
sweep["resample"] = {k: {kk: round(vv, 4) for kk, vv in rs[k].items() if isinstance(vv, (int, float))}
                     for k in ("delta_vs_kq", "delta_vs_uniform")}
sweep["original"] = {k: round(v, 4) for k, v in cs["original_sample"].items()}

data = {
    "num": num,
    "calibration": calibration,
    "regret": regret,
    "resolution": resolution,
    "sleep": sleep,
    "af_raw": af_raw,
    "af_axis": af_curve,
    "design_methods": design_methods,
    "fig1": fig1,
    "sweep": sweep,
}
OUT.write_text("// Generated by tools/extract_data.py -- do not edit by hand.\n"
               "window.CPV_DATA = " + json.dumps(data, indent=1) + ";\n")
print("wrote", OUT, OUT.stat().st_size, "bytes")
print("af keys:", list(af_raw)[:40])
print("sleep methods:", {k: list(v) for k, v in sleep.items()})
