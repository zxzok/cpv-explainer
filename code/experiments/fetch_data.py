"""Acquire two real full-trajectory datasets and convert them to per-subject
discrete state series.

(1) Sleep-EDF Expanded hypnograms (PhysioNet ``sleep-edfx/1.0.0``).
    Only the tiny ``*-Hypnogram.edf`` annotation files are downloaded -- never
    the multi-gigabyte ``*-PSG.edf`` signal files.  Each recording becomes a
    30-second-epoch stage sequence over
    ``{W, N1, N2, N3, REM, MOVEMENT, UNKNOWN}``.

(2) Long-Term AF Database (PhysioNet ``ltafdb/1.0.0``).  Each record becomes a
    60-second-epoch "in AF or not" series obtained by forward-filling the
    rhythm annotations at 1 s resolution and averaging into minutes.

Outputs
-------
``data/sleep_edf/hypnograms.npz``  stages/subject_ids/record_ids/...
``data/sleep_edf/meta.json``
``data/ltaf/af_series.npz``        af/record_ids/...
``data/ltaf/meta.json``
``results/data_summary.json``      counts, median lengths, label marginals
``results/data_summary.csv``       one row per recording
``figures/data_overview.pdf/.png`` example hypnograms + AF-burden histogram

Run with::

    .venv/bin/python experiments/fetch_data.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import (DATA, RESULTS, SEED, Timer, environment_record,
                                PALETTE, save_csv, save_figure, save_json,
                                setup_matplotlib)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SLEEP_BASE = "https://physionet.org/files/sleep-edfx/1.0.0"
ST_SUBJECTS_URL = f"{SLEEP_BASE}/ST-subjects.xls"
LTAF_BASE = "https://physionet.org/files/ltafdb/1.0.0"
SLEEP_SUBDIRS = ("sleep-cassette", "sleep-telemetry")

EPOCH_SLEEP = 30      # seconds per hypnogram epoch
EPOCH_LTAF = 60       # seconds per AF epoch
TRIM_PAD_EPOCHS = 60  # 30 min of wake padding kept either side of the sleep period
N_DOWNLOAD_WORKERS = 8
HTTP_TIMEOUT = 120

SLEEP_DIR = DATA / "sleep_edf"
SLEEP_RAW = SLEEP_DIR / "raw"
LTAF_DIR = DATA / "ltaf"
LTAF_RAW = DATA / "ltafdb"
for _d in (SLEEP_DIR, SLEEP_RAW, LTAF_DIR, LTAF_RAW):
    _d.mkdir(parents=True, exist_ok=True)

# Integer codes stored in the npz (int8).
STAGE_MAP = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "MOVEMENT": 5,
    "UNKNOWN": 6,
}
STAGE_NAMES = {v: k for k, v in STAGE_MAP.items()}
SLEEP_CODES = (STAGE_MAP["N1"], STAGE_MAP["N2"], STAGE_MAP["N3"], STAGE_MAP["REM"])
VALID_SLEEP_STAGE_CODES = (STAGE_MAP["W"],) + SLEEP_CODES

# EDF+ annotation strings used by Sleep-EDF Expanded.  "Sleep stage 4" is the
# old R&K stage 4 and is merged into N3 (AASM).
DESCRIPTION_MAP = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage 2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "sleep stage r": "REM",
    "movement time": "MOVEMENT",
    "sleep stage ?": "UNKNOWN",
}

AF_RHYTHMS = {"AFIB", "AFL"}


# --------------------------------------------------------------------------
# HTTP helpers (with caching on disk so re-runs are cheap)
# --------------------------------------------------------------------------
def _session():
    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:                                    # pragma: no cover
        from requests.packages.urllib3.util.retry import Retry
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=0.6,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET", "HEAD"]))
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=32))
    s.headers.update({"User-Agent": "protocol-ceiling-experiments/0.1"})
    return s


def _get_text(sess, url: str) -> str:
    r = sess.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text


def _download(sess, url: str, dest: Path, min_bytes: int = 64) -> Path:
    """Download ``url`` to ``dest`` unless a plausible copy already exists."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    r = sess.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------
# EDF+ annotation parsing
# --------------------------------------------------------------------------
def parse_edf_annotations_manual(path: Path):
    """Hand parser for EDF+ TAL records -> (onsets, durations, descriptions).

    EDF header = 256 global bytes + 256 bytes per signal.  Each data record
    stores, for the "EDF Annotations" signal, a sequence of TALs of the form
    ``onset \\x15 duration \\x14 description \\x14 ... \\x00``.
    """
    raw = path.read_bytes()
    if len(raw) < 256:
        raise ValueError("file shorter than an EDF header")
    n_records = int(raw[236:244].decode("ascii", "ignore").strip())
    n_signals = int(raw[252:256].decode("ascii", "ignore").strip())
    hdr_len = 256 + 256 * n_signals
    if len(raw) < hdr_len:
        raise ValueError("truncated EDF header")

    off = 256
    labels = [raw[off + 16 * i: off + 16 * (i + 1)].decode("ascii", "ignore").strip()
              for i in range(n_signals)]
    off += 16 * n_signals            # labels
    off += 80 * n_signals            # transducer
    off += 8 * n_signals             # physical dimension
    off += 8 * n_signals             # physical minimum
    off += 8 * n_signals             # physical maximum
    off += 8 * n_signals             # digital minimum
    off += 8 * n_signals             # digital maximum
    off += 80 * n_signals            # prefiltering
    n_samps = [int(raw[off + 8 * i: off + 8 * (i + 1)].decode("ascii", "ignore").strip())
               for i in range(n_signals)]

    ann_idx = [i for i, lab in enumerate(labels) if lab == "EDF Annotations"]
    if not ann_idx:
        raise ValueError("no 'EDF Annotations' signal in header")

    bytes_per_sig = [2 * n for n in n_samps]
    rec_len = sum(bytes_per_sig)
    starts = np.cumsum([0] + bytes_per_sig[:-1])

    onsets, durations, descriptions = [], [], []
    for r in range(n_records):
        base = hdr_len + r * rec_len
        if base + rec_len > len(raw):
            break
        for si in ann_idx:
            blob = raw[base + starts[si]: base + starts[si] + bytes_per_sig[si]]
            for tal in blob.split(b"\x00"):
                if not tal:
                    continue
                fields = tal.split(b"\x14")
                head = fields[0]
                if b"\x15" in head:
                    on_b, dur_b = head.split(b"\x15", 1)
                else:
                    on_b, dur_b = head, b""
                try:
                    onset = float(on_b.decode("ascii", "ignore"))
                except ValueError:
                    continue
                try:
                    dur = float(dur_b.decode("ascii", "ignore")) if dur_b.strip() else 0.0
                except ValueError:
                    dur = 0.0
                for desc in fields[1:]:
                    text = desc.decode("latin-1").strip()
                    if not text:
                        continue
                    onsets.append(onset)
                    durations.append(dur)
                    descriptions.append(text)
    if not onsets:
        raise ValueError("no annotations found")
    return np.asarray(onsets), np.asarray(durations), descriptions


def parse_edf_annotations(path: Path, prefer_mne: bool = True):
    """(a) mne if importable, else (b) the hand parser."""
    if prefer_mne:
        try:
            import mne
            mne.set_log_level("ERROR")
            ann = mne.read_annotations(str(path))
            return (np.asarray(ann.onset, float), np.asarray(ann.duration, float),
                    [str(d) for d in ann.description], "mne")
        except Exception:
            pass
    on, dur, desc = parse_edf_annotations_manual(path)
    return on, dur, desc, "manual"


def hypnogram_from_annotations(onsets, durations, descriptions):
    """Rasterise EDF+ annotations onto a 30 s epoch grid of int8 stage codes."""
    total = 0.0
    for o, d in zip(onsets, durations):
        total = max(total, o + d)
    n_epochs = int(round(total / EPOCH_SLEEP))
    if n_epochs <= 0:
        raise ValueError("empty annotation span")
    stages = np.full(n_epochs, STAGE_MAP["UNKNOWN"], dtype=np.int8)

    unknown_desc = set()
    for onset, dur, desc in zip(onsets, durations, descriptions):
        if dur <= 0:
            continue
        key = str(desc).strip().lower()
        name = DESCRIPTION_MAP.get(key)
        if name is None:
            unknown_desc.add(str(desc).strip())
            name = "UNKNOWN"
        i0 = int(round(onset / EPOCH_SLEEP))
        i1 = int(round((onset + dur) / EPOCH_SLEEP))
        i0 = max(0, min(n_epochs, i0))
        i1 = max(0, min(n_epochs, i1))
        if i1 > i0:
            stages[i0:i1] = STAGE_MAP[name]
    return stages, sorted(unknown_desc)


def trim_wake_padding(stages: np.ndarray):
    """Keep 30 min either side of the first/last non-Wake epoch."""
    is_sleep = np.isin(stages, SLEEP_CODES)
    if not is_sleep.any():
        return None
    first, last = int(np.argmax(is_sleep)), int(len(is_sleep) - 1 - np.argmax(is_sleep[::-1]))
    lo = max(0, first - TRIM_PAD_EPOCHS)
    hi = min(len(stages), last + 1 + TRIM_PAD_EPOCHS)
    return stages[lo:hi], first, last


def sleep_edf_identifiers(record_id: str):
    """``SC4ssNEO`` / ``ST7ssNJ0`` -> (cohort, subject_id, night)."""
    cohort = record_id[:2]                      # 'SC' or 'ST'
    subject = f"{cohort}{record_id[3:5]}"       # subject digits ss
    night = record_id[5]
    return cohort, subject, night


# --------------------------------------------------------------------------
# Dataset 1: Sleep-EDF Expanded hypnograms
# --------------------------------------------------------------------------
def list_hypnogram_files(sess) -> list[tuple[str, str]]:
    """Return [(subdir, filename), ...] for every ``*-Hypnogram.edf``."""
    out = []
    for sub in SLEEP_SUBDIRS:
        html = _get_text(sess, f"{SLEEP_BASE}/{sub}/")
        names = sorted(set(re.findall(r"[A-Z]{2}\d[0-9A-Za-z]*-Hypnogram\.edf", html)))
        out.extend((sub, n) for n in names)
        print(f"  {sub}: {len(names)} hypnogram files listed")
    return out


def sleep_telemetry_treatments(path: Path) -> dict[tuple[str, str], str]:
    """Read the placebo/temazepam night assignment from PhysioNet metadata."""
    import xlrd

    sheet = xlrd.open_workbook(path).sheet_by_index(0)
    out: dict[tuple[str, str], str] = {}
    for row in range(2, sheet.nrows):
        value = sheet.cell_value(row, 0)
        if value in (None, ""):
            continue
        subject = f"ST{int(value):02d}"
        placebo_night = str(int(sheet.cell_value(row, 3)))
        temazepam_night = str(int(sheet.cell_value(row, 5)))
        if placebo_night == temazepam_night or {placebo_night, temazepam_night} != {"1", "2"}:
            raise ValueError(f"invalid ST treatment-night assignment for {subject}")
        out[(subject, placebo_night)] = "placebo"
        out[(subject, temazepam_night)] = "temazepam"
    if len(out) != 44:
        raise ValueError(f"expected 44 ST recording assignments, found {len(out)}")
    return out


def build_sleep_edf(sess) -> tuple[dict, list[dict], list[str]]:
    files = list_hypnogram_files(sess)
    if not files:
        raise RuntimeError("no hypnogram files found in the directory listings")

    def _fetch(item):
        sub, name = item
        return _download(sess, f"{SLEEP_BASE}/{sub}/{name}", SLEEP_RAW / name)

    with ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS) as ex:
        list(ex.map(_fetch, files))
    st_subjects_path = _download(sess, ST_SUBJECTS_URL,
                                 SLEEP_DIR / "ST-subjects.xls", min_bytes=1024)
    st_treatment = sleep_telemetry_treatments(st_subjects_path)
    total_mb = sum((SLEEP_RAW / n).stat().st_size for _, n in files) / 1e6
    print(f"  downloaded/cached {len(files)} hypnograms ({total_mb:.2f} MB)")

    stages_list, oracle_stages_list, rows, failures = [], [], [], []
    parsers, extra_desc = {}, set()
    cross_epochs, cross_agree, cross_disagree = [0], [0], []
    for sub, name in files:
        rid = name.replace("-Hypnogram.edf", "")
        path = SLEEP_RAW / name
        try:
            on, dur, desc, how = parse_edf_annotations(path)
            parsers[how] = parsers.get(how, 0) + 1
            raw_stages, unk = hypnogram_from_annotations(on, dur, desc)
            extra_desc.update(unk)
            # Cross-check, not fallback: when MNE parsed the file, rasterise the
            # self-contained TAL parser's output too and require the two stage
            # vectors to agree epoch for epoch.  A silent disagreement between
            # the two readers would change every downstream number, so it is
            # asserted here rather than assumed.
            if how == "mne":
                on2, dur2, desc2 = parse_edf_annotations_manual(path)
                alt, _ = hypnogram_from_annotations(on2, dur2, desc2)
                n = min(alt.size, raw_stages.size)
                agree = int(np.sum(alt[:n] == raw_stages[:n]))
                cross_epochs[0] += int(max(alt.size, raw_stages.size))
                cross_agree[0] += agree
                if alt.size != raw_stages.size or agree != n:
                    cross_disagree.append(
                        f"{rid}: mne {raw_stages.size} epochs vs manual "
                        f"{alt.size}, {n - agree} differing epochs")
            trimmed = trim_wake_padding(raw_stages)
            if trimmed is None:
                failures.append(f"{rid}: no non-Wake epoch, skipped")
                continue
            oracle_stages, first, last = trimmed
            # Primary interval: complete EDF annotation interval, independent of
            # the realised sleep labels. Movement and unknown epochs are removed
            # because they are not valid stage scores. The first/last-non-Wake
            # alignment is retained as an explicitly oracle-aligned sensitivity.
            stages = raw_stages[np.isin(raw_stages, VALID_SLEEP_STAGE_CODES)]
            if stages.size < 128:
                failures.append(f"{rid}: fewer than 128 valid full-record epochs, skipped")
                continue
        except Exception as exc:                            # noqa: BLE001
            failures.append(f"{rid}: {type(exc).__name__}: {exc}")
            continue

        cohort, subject, night = sleep_edf_identifiers(rid)
        treatment = "none" if cohort == "SC" else st_treatment[(subject, night)]
        counts = np.bincount(stages.astype(int), minlength=len(STAGE_MAP))
        row = {
            "dataset": "sleep_edf",
            "record_id": rid,
            "subject_id": subject,
            "cohort": cohort,
            "night": night,
            "treatment": treatment,
            "n_epochs": int(stages.size),
            "n_epochs_oracle_aligned": int(oracle_stages.size),
            "epoch_seconds": EPOCH_SLEEP,
            "hours": float(stages.size * EPOCH_SLEEP / 3600.0),
            "oracle_aligned_hours": float(oracle_stages.size * EPOCH_SLEEP / 3600.0),
            "raw_hours": float(raw_stages.size * EPOCH_SLEEP / 3600.0),
            "sleep_period_hours": float((last - first + 1) * EPOCH_SLEEP / 3600.0),
            "total_sleep_hours": float(np.isin(stages, SLEEP_CODES).sum()
                                       * EPOCH_SLEEP / 3600.0),
        }
        for code, nm in STAGE_NAMES.items():
            row[f"frac_{nm}"] = float(counts[code] / counts.sum())
        rows.append(row)
        stages_list.append(stages)
        oracle_stages_list.append(oracle_stages)

    stages_obj = np.empty(len(stages_list), dtype=object)
    stages_obj[:] = stages_list
    oracle_obj = np.empty(len(oracle_stages_list), dtype=object)
    oracle_obj[:] = oracle_stages_list
    record_ids = np.array([r["record_id"] for r in rows])
    subject_ids = np.array([r["subject_id"] for r in rows])
    cohorts = np.array([r["cohort"] for r in rows])
    nights = np.array([r["night"] for r in rows])
    treatments = np.array([r["treatment"] for r in rows])

    npz_path = SLEEP_DIR / "hypnograms.npz"
    np.savez(npz_path, stages=stages_obj, stages_full=stages_obj,
             stages_oracle_aligned=oracle_obj, subject_ids=subject_ids,
             record_ids=record_ids, cohorts=cohorts, nights=nights,
             treatments=treatments,
             epoch_seconds=np.int64(EPOCH_SLEEP))
    print(f"  [npz]  {npz_path.relative_to(DATA.parent)} "
          f"({len(rows)} recordings, {len(set(subject_ids))} subjects)")

    meta = {
        "source": SLEEP_BASE,
        "subdirectories": list(SLEEP_SUBDIRS),
        "downloaded_files": ("*-Hypnogram.edf and ST-subjects.xls metadata only "
                             "(no PSG signal files)"),
        "treatment_metadata": {
            "source": ST_SUBJECTS_URL,
            "columns": "Placebo night and Temazepam night",
            "n_record_assignments": len(st_treatment),
        },
        "epoch_seconds": EPOCH_SLEEP,
        "stage_map": STAGE_MAP,
        "stage_names": {str(k): v for k, v in STAGE_NAMES.items()},
        "description_map": DESCRIPTION_MAP,
        "merge_note": "R&K 'Sleep stage 4' merged into N3; "
                      "'Movement time' -> MOVEMENT; 'Sleep stage ?' -> UNKNOWN",
        "primary_interval_rule": ("complete EDF annotation interval; movement and "
                                  "unknown epochs removed from the valid-stage sequence"),
        "oracle_sensitivity_rule": ("kept from 30 min (60 epochs) before the first "
                                    "non-Wake epoch to 30 min after the last non-Wake epoch"),
        "record_id_format": "SC4ssNEO / ST7ssNJ0; subject = cohort + ss, night = N",
        "n_recordings": len(rows),
        "n_subjects": int(len(set(subject_ids))),
        "parser_used": parsers,
        "parser_cross_check": {
            "epochs_compared": cross_epochs[0],
            "epochs_agreeing": cross_agree[0],
            "agreement": (cross_agree[0] / cross_epochs[0]) if cross_epochs[0] else None,
            "records_disagreeing": len(cross_disagree),
            "examples": cross_disagree[:5],
            "note": ("every hypnogram MNE parsed was also parsed by the "
                     "self-contained EDF+ TAL reader in this file and the two "
                     "rasterised stage vectors compared epoch by epoch"),
        },
        "unmapped_descriptions": sorted(extra_desc),
        "failures": failures,
        "npz_note": "load with np.load(path, allow_pickle=True); "
                    "'stages' is an object array of int8 arrays",
        "seed": SEED,
    }
    (SLEEP_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [json] data/sleep_edf/meta.json")
    if failures:
        print(f"  !! {len(failures)} recordings skipped")
        for f in failures[:10]:
            print(f"     {f}")
    return meta, rows, list(stages_list)


# --------------------------------------------------------------------------
# Dataset 2: Long-Term AF Database
# --------------------------------------------------------------------------
def build_ltaf(sess) -> tuple[dict, list[dict], list[np.ndarray]]:
    import wfdb

    try:
        records = [str(r).strip("/") for r in wfdb.get_record_list("ltafdb")]
    except Exception:                                        # pragma: no cover
        html = _get_text(sess, f"{LTAF_BASE}/")
        records = sorted({m for m in re.findall(r">(\d+)\.hea<", html)})
    print(f"  {len(records)} LTAF records listed")

    def _fetch(rec):
        for ext in (".hea", ".atr"):
            _download(sess, f"{LTAF_BASE}/{rec}{ext}", LTAF_RAW / f"{rec}{ext}")
        return rec

    with ThreadPoolExecutor(max_workers=N_DOWNLOAD_WORKERS) as ex:
        list(ex.map(_fetch, records))
    total_mb = sum(p.stat().st_size for p in LTAF_RAW.glob("*")) / 1e6
    print(f"  downloaded/cached {len(records)} .hea/.atr pairs ({total_mb:.1f} MB)")

    af_list, rows, failures = [], [], []
    rhythm_counter: dict[str, int] = {}
    for rec in records:
        stem = str(LTAF_RAW / rec)
        try:
            hdr = wfdb.rdheader(stem)
            fs = float(hdr.fs)
            n_sec = int(hdr.sig_len // fs)
            ann = wfdb.rdann(stem, "atr")
        except Exception as exc:                             # noqa: BLE001
            failures.append(f"{rec}: {type(exc).__name__}: {exc}")
            continue
        if n_sec < 60:
            failures.append(f"{rec}: shorter than one epoch")
            continue

        notes = [str(a).replace("\x00", "").strip() for a in ann.aux_note]
        idx = [i for i, s in enumerate(notes) if s.startswith("(")]
        bounds, flags, labels = [], [], []
        for i in idx:
            label = notes[i][1:].strip()
            labels.append(label)
            rhythm_counter[label] = rhythm_counter.get(label, 0) + 1
            bounds.append(int(np.clip(int(ann.sample[i]) / fs, 0, n_sec)))
            flags.append(1 if label.upper() in AF_RHYTHMS else 0)

        series_1s = np.zeros(n_sec, dtype=np.int8)
        for k, b0 in enumerate(bounds):
            b1 = bounds[k + 1] if k + 1 < len(bounds) else n_sec
            if b1 > b0:
                series_1s[b0:b1] = flags[k]
        prefix = bounds[0] if bounds else n_sec

        # Seconds before the first rhythm marker carry no rhythm evidence.
        # Coding them as non-AF would silently change the estimand, which is
        # time-in-AF divided by *analysable* recording time, so the analysable
        # window starts at the first marker instead.
        series_1s = series_1s[prefix:]
        n_analysable = series_1s.size

        n_ep = n_analysable // EPOCH_LTAF
        if n_ep < 1:
            failures.append(f"{rec}: no analysable span after the prefix rule")
            continue
        af60 = series_1s[: n_ep * EPOCH_LTAF].astype(np.float64)
        af60 = af60.reshape(n_ep, EPOCH_LTAF).mean(axis=1)

        af_list.append(af60.astype(np.float32))
        rows.append({
            "dataset": "ltaf",
            "record_id": rec,
            "subject_id": rec,
            "n_epochs": int(n_ep),
            "epoch_seconds": EPOCH_LTAF,
            "hours": float(n_ep * EPOCH_LTAF / 3600.0),
            "analysable_hours": float(n_ep * EPOCH_LTAF / 3600.0),
            "raw_hours": float(n_sec / 3600.0),
            "fs_hz": fs,
            "n_beats": int(len(ann.sample)),
            "n_rhythm_markers": int(len(idx)),
            "af_burden": float(af60.mean()),
            "frac_af_seconds": float(series_1s.mean()),
            "unlabelled_prefix_hours": float(prefix / 3600.0),
            "n_af_episodes": int(np.sum(np.diff(np.concatenate(
                ([0], series_1s.astype(int), [0]))) == 1)),
        })

    af_obj = np.empty(len(af_list), dtype=object)
    af_obj[:] = af_list
    record_ids = np.array([r["record_id"] for r in rows])
    npz_path = LTAF_DIR / "af_series.npz"
    np.savez(npz_path, af=af_obj, record_ids=record_ids,
             epoch_seconds=np.int64(EPOCH_LTAF))
    print(f"  [npz]  data/ltaf/af_series.npz ({len(rows)} records)")

    meta = {
        "source": LTAF_BASE,
        "epoch_seconds": EPOCH_LTAF,
        "construction": ("rhythm annotations forward-filled from each change "
                         "point to the next at 1 s resolution, then averaged "
                         "into 60 s epochs"),
        "af_rhythms": sorted(AF_RHYTHMS),
        "value_semantics": "fraction of the 60 s epoch spent in AF/AFL, in [0, 1]",
        "prefix_rule": ("seconds before the first rhythm marker carry no rhythm "
                        "evidence and are dropped; the analysable window starts "
                        "at the first marker"),
        "rhythm_marker_counts": dict(sorted(rhythm_counter.items(),
                                            key=lambda kv: -kv[1])),
        "n_records": len(rows),
        "failures": failures,
        "npz_note": "load with np.load(path, allow_pickle=True); "
                    "'af' is an object array of float32 arrays",
        "seed": SEED,
    }
    (LTAF_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  [json] data/ltaf/meta.json")
    if failures:
        print(f"  !! {len(failures)} records skipped")
        for f in failures[:10]:
            print(f"     {f}")
    return meta, rows, af_list


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
# Conventional hypnogram ordering: W on top, N3 at the bottom.
PLOT_LEVEL = {STAGE_MAP["W"]: 4, STAGE_MAP["REM"]: 3, STAGE_MAP["N1"]: 2,
              STAGE_MAP["N2"]: 1, STAGE_MAP["N3"]: 0}


def make_figure(sleep_rows, sleep_stages, ltaf_rows, rng):
    plt = setup_matplotlib()
    cassette = [i for i, r in enumerate(sleep_rows) if r["cohort"] == "SC"]
    pick = sorted(rng.choice(cassette, size=min(3, len(cassette)), replace=False))

    fig = plt.figure(figsize=(7.0, 3.1))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1.0], wspace=0.28, hspace=0.28)

    axes = [fig.add_subplot(gs[k, 0]) for k in range(3)]
    t_max = max(sleep_stages[i].size for i in pick) * EPOCH_SLEEP / 3600.0
    for ax, i, colour in zip(axes, pick, PALETTE):
        st = sleep_stages[i]
        lvl = np.array([PLOT_LEVEL.get(int(s), np.nan) for s in st], dtype=float)
        t = np.arange(st.size) * EPOCH_SLEEP / 3600.0
        ax.step(t, lvl, where="post", color=colour, lw=0.8,
                label=f"{sleep_rows[i]['record_id']} ({sleep_rows[i]['hours']:.1f} h)")
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["N3", "N2", "N1", "R", "W"])
        ax.set_ylim(-0.5, 4.5)
        ax.set_xlim(0, t_max)
        leg = ax.legend(loc="lower right", handlelength=1.2, borderpad=0.15,
                        frameon=True, framealpha=0.92, edgecolor="none")
        leg.get_frame().set_facecolor("white")
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("Time through complete valid-stage sequence (h)")
    axes[1].set_ylabel("Sleep stage")

    ax2 = fig.add_subplot(gs[:, 1])
    burden = np.array([r["af_burden"] for r in ltaf_rows])
    ax2.hist(burden, bins=np.linspace(0, 1, 21), color=PALETTE[0],
             edgecolor="white", lw=0.5, label=f"LTAF records (n={burden.size})")
    ax2.axvline(float(np.median(burden)), color=PALETTE[1], ls="--", lw=1.2,
                label=f"median = {np.median(burden):.2f}")
    frac_mixed = float(np.mean((burden > 0) & (burden < 1)))
    ax2.plot([], [], ls="none", label=f"$0<b<1$: {100 * frac_mixed:.0f}%")
    ax2.set_xlabel("AF burden $b$ (fraction of 60 s epochs in AF)")
    ax2.set_ylabel("Number of records")
    ax2.set_xlim(0, 1)
    ax2.legend(loc="upper center", handlelength=1.2)

    save_figure(fig, "data_overview")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sess = _session()

    with Timer("sleep-edf"):
        sleep_meta, sleep_rows, sleep_stages = build_sleep_edf(sess)
    with Timer("ltafdb"):
        ltaf_meta, ltaf_rows, _af = build_ltaf(sess)

    # ---- summaries -------------------------------------------------------
    s_hours = np.array([r["hours"] for r in sleep_rows])
    s_spt = np.array([r["sleep_period_hours"] for r in sleep_rows])
    s_raw = np.array([r["raw_hours"] for r in sleep_rows])
    cas = np.array([r["cohort"] == "SC" for r in sleep_rows])
    all_stages = np.concatenate(sleep_stages)
    counts = np.bincount(all_stages.astype(int), minlength=len(STAGE_MAP))
    marginal = {STAGE_NAMES[c]: float(counts[c] / counts.sum())
                for c in range(len(STAGE_MAP))}

    burden = np.array([r["af_burden"] for r in ltaf_rows])
    l_hours = np.array([r["hours"] for r in ltaf_rows])
    frac_mixed = float(np.mean((burden > 0) & (burden < 1)))

    summary = {
        "seed": SEED,
        "environment": environment_record(),
        "sleep_edf": {
            "source": SLEEP_BASE,
            "n_recordings": int(len(sleep_rows)),
            "n_subjects": int(len({r["subject_id"] for r in sleep_rows})),
            "n_cassette_recordings": int(cas.sum()),
            "n_telemetry_recordings": int((~cas).sum()),
            "n_cassette_subjects": int(len({r["subject_id"] for r in sleep_rows
                                            if r["cohort"] == "SC"})),
            "n_telemetry_subjects": int(len({r["subject_id"] for r in sleep_rows
                                             if r["cohort"] == "ST"})),
            "epoch_seconds": EPOCH_SLEEP,
            "n_epochs_total": int(all_stages.size),
            "total_hours": float(s_hours.sum()),
            "total_hours_before_trim": float(s_raw.sum()),
            "median_record_hours": float(np.median(s_hours)),
            "iqr_record_hours": [float(np.percentile(s_hours, 25)),
                                 float(np.percentile(s_hours, 75))],
            "median_sleep_period_hours": float(np.median(s_spt)),
            "median_sleep_period_hours_cassette": float(np.median(s_spt[cas])),
            "median_sleep_period_hours_telemetry": float(np.median(s_spt[~cas])),
            "median_record_hours_cassette": float(np.median(s_hours[cas])),
            "median_record_hours_telemetry": float(np.median(s_hours[~cas])),
            "marginal_label_distribution": marginal,
            "n_records_over_12h": int(np.sum(s_hours > 12.0)),
            "n_failed": len(sleep_meta["failures"]),
            "parser_used": sleep_meta["parser_used"],
        },
        "ltaf": {
            "source": LTAF_BASE,
            "n_records": int(len(ltaf_rows)),
            "epoch_seconds": EPOCH_LTAF,
            "n_epochs_total": int(sum(r["n_epochs"] for r in ltaf_rows)),
            "total_hours": float(l_hours.sum()),
            "median_record_hours": float(np.median(l_hours)),
            "iqr_record_hours": [float(np.percentile(l_hours, 25)),
                                 float(np.percentile(l_hours, 75))],
            "marginal_label_distribution": {
                "AF": float(np.average(burden, weights=[r["n_epochs"] for r in ltaf_rows])),
                "non_AF": float(1.0 - np.average(
                    burden, weights=[r["n_epochs"] for r in ltaf_rows])),
            },
            "mean_af_burden_per_record": float(burden.mean()),
            "median_af_burden_per_record": float(np.median(burden)),
            "n_records_burden_zero": int(np.sum(burden <= 0.0)),
            "n_records_burden_one": int(np.sum(burden >= 1.0)),
            "n_records_burden_strictly_between": int(np.sum((burden > 0) & (burden < 1))),
            "frac_records_burden_strictly_between": frac_mixed,
            # prefix statistics excluding the single longest prefix, which is the
            # record the sensitivity analysis isolates
            "unlabelled_prefix_seconds_median_excluding_longest": float(np.median(
                np.sort([r["unlabelled_prefix_hours"] * 3600.0 for r in ltaf_rows])[:-1])),
            "unlabelled_prefix_seconds_max_excluding_longest": float(
                np.sort([r["unlabelled_prefix_hours"] * 3600.0 for r in ltaf_rows])[-2]),
            "n_records_excluding_longest_prefix": int(len(ltaf_rows) - 1),
            "n_records_burden_in_0p05_0p95": int(np.sum((burden > 0.05) & (burden < 0.95))),
            "frac_records_burden_in_0p05_0p95": float(np.mean((burden > 0.05)
                                                             & (burden < 0.95))),
            "n_failed": len(ltaf_meta["failures"]),
        },
        "sanity_checks": {
            "sleep_edf_total_recording_hours": float(s_hours.sum()),
            "ltaf_total_recording_hours": float(l_hours.sum()),
            "median_sleep_period_hours_cassette": float(np.median(s_spt[cas])),
            "median_trimmed_record_hours_cassette": float(np.median(s_hours[cas])),
            "median_sleep_period_in_7_to_9h": bool(
                7.0 <= float(np.median(s_spt[cas])) <= 9.0),
            "median_trimmed_record_in_7_to_9h": bool(
                7.0 <= float(np.median(s_hours[cas])) <= 9.0),
            "frac_ltaf_records_burden_strictly_between_0_and_1": frac_mixed,
            "ltaf_mixed_is_clear_majority": bool(frac_mixed > 0.5),
            "sleep_edf_at_least_150_recordings": bool(len(sleep_rows) >= 150),
        },
        "outputs": {
            "sleep_npz": str((SLEEP_DIR / "hypnograms.npz").resolve()),
            "sleep_meta": str((SLEEP_DIR / "meta.json").resolve()),
            "ltaf_npz": str((LTAF_DIR / "af_series.npz").resolve()),
            "ltaf_meta": str((LTAF_DIR / "meta.json").resolve()),
        },
    }
    summary["runtime_seconds"] = float(time.perf_counter() - t0)

    # One CSV row per recording, unified schema across the two datasets.
    keys: list[str] = []
    for r in sleep_rows + ltaf_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    save_csv([{k: r.get(k, "") for k in keys} for r in sleep_rows + ltaf_rows],
             "data_summary")
    save_json(summary, "data_summary")
    make_figure(sleep_rows, sleep_stages, ltaf_rows, rng)

    sc = summary["sanity_checks"]
    print("\nSANITY CHECKS")
    for k, v in sc.items():
        print(f"  {k}: {v}")
    print(f"\ntotal runtime {summary['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
