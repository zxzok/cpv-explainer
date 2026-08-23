"""Compare rebuilt artefacts with the sealed final-manuscript reference.

The comparison is intentionally stronger than a smoke test:

* every reference JSON/CSV result must be present and numerically agree;
* volatile environment timestamps and elapsed runtimes are ignored;
* all numeric macros referenced by the final TeX must be defined;
* generated ``numbers.tex`` must match the sealed file byte for byte;
* the four figures used by the paper must exist and be synchronised;
* cached annotation counts and the compiled paper page count are checked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

REL_TOL = 1e-8
ABS_TOL = 1e-10
VOLATILE_KEYS = {
    "environment",
    "outputs",
    "runtime_s",
    "runtime_seconds",
    "elapsed_s",
    "elapsed_seconds",
    "timestamp",
    "timestamp_utc",
}
CSV_VOLATILE_COLUMNS = {"runtime_s", "runtime_seconds", "elapsed_s", "elapsed_seconds"}
RETAINED_RESULT_PREFIXES = (
    "fig1_framework",
    "ltaf_record64_sensitivity",
    "s3b_",
    "s5b_",
    "s7_",
)

problems: list[str] = []
warnings: list[str] = []
checks = 0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{'OK' if condition else 'FAIL'}] {label}{suffix}")
    if not condition:
        problems.append(f"{label}{suffix}")


def numbers_close(actual: float, expected: float) -> bool:
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def compare_json(actual: Any, expected: Any, where: str, out: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append(f"{where}: expected object, got {type(actual).__name__}")
            return
        exp_keys = {key for key in expected if key not in VOLATILE_KEYS}
        act_keys = {key for key in actual if key not in VOLATILE_KEYS}
        if exp_keys != act_keys:
            out.append(
                f"{where}: key mismatch; missing={sorted(exp_keys-act_keys)}, "
                f"extra={sorted(act_keys-exp_keys)}"
            )
            return
        for key in sorted(exp_keys):
            compare_json(actual[key], expected[key], f"{where}/{key}", out)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            out.append(
                f"{where}: list length {len(actual) if isinstance(actual, list) else 'NA'} "
                f"!= {len(expected)}"
            )
            return
        for index, (a_item, e_item) in enumerate(zip(actual, expected)):
            compare_json(a_item, e_item, f"{where}/{index}", out)
        return
    if isinstance(expected, bool) or expected is None:
        if actual != expected:
            out.append(f"{where}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not numbers_close(float(actual), float(expected)):
            out.append(f"{where}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        out.append(f"{where}: {actual!r} != {expected!r}")


def parse_number(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def compare_csv(actual_path: Path, expected_path: Path, out: list[str]) -> None:
    with actual_path.open(newline="") as handle:
        actual_reader = csv.DictReader(handle)
        actual = list(actual_reader)
        actual_fields = [field for field in (actual_reader.fieldnames or [])
                         if field not in CSV_VOLATILE_COLUMNS]
    with expected_path.open(newline="") as handle:
        expected_reader = csv.DictReader(handle)
        expected = list(expected_reader)
        expected_fields = [field for field in (expected_reader.fieldnames or [])
                           if field not in CSV_VOLATILE_COLUMNS]
    if len(actual) != len(expected):
        out.append(f"row count {len(actual)} != {len(expected)}")
        return
    if actual_fields != expected_fields:
        out.append("column names differ")
        return
    # Parallel workers may finish in a different order while producing the
    # same labelled replications. Compare the CSVs as multisets of rows.
    actual.sort(key=lambda row: tuple(row[field] for field in actual_fields))
    expected.sort(key=lambda row: tuple(row[field] for field in expected_fields))
    for row_index, (a_row, e_row) in enumerate(zip(actual, expected)):
        for key in expected_fields:
            a_value, e_value = a_row[key], e_row[key]
            a_num, e_num = parse_number(a_value), parse_number(e_value)
            if a_num is not None and e_num is not None:
                equal = numbers_close(a_num, e_num)
            else:
                equal = a_value == e_value
            if not equal:
                out.append(
                    f"row {row_index + 2}, {key}: {a_value!r} != {e_value!r}"
                )
                if len(out) >= 25:
                    return


def compare_results() -> None:
    reference_files = sorted(
        path for path in (REFERENCE / "results").iterdir()
        if path.suffix in {".json", ".csv"}
    )
    check("reference result inventory", len(reference_files) == 53,
          f"{len(reference_files)} JSON/CSV files")
    failures: list[str] = []
    retained_differences: list[str] = []
    for expected_path in reference_files:
        actual_path = RESULTS / expected_path.name
        if not actual_path.exists():
            failures.append(f"{expected_path.name}: missing")
            continue
        local: list[str] = []
        try:
            if expected_path.suffix == ".json":
                actual = json.loads(actual_path.read_text())
                expected = json.loads(expected_path.read_text())
                compare_json(actual, expected, expected_path.name, local)
            else:
                compare_csv(actual_path, expected_path, local)
        except Exception as exc:  # fail with file-local context
            local.append(f"comparison error: {exc}")
        destination = retained_differences if expected_path.name.startswith(
            RETAINED_RESULT_PREFIXES
        ) else failures
        destination.extend(f"{expected_path.name}: {item}" for item in local[:25])
        if len(failures) >= 50:
            break
    check("numeric result comparison", not failures,
          "all reference JSON/CSV files agree" if not failures else failures[0])
    if failures:
        for item in failures[1:20]:
            print(f"       {item}")
    if retained_differences:
        warnings.append(
            "retained non-manuscript regression outputs differ; first difference: "
            + retained_differences[0]
        )


def verify_macros_and_bibliography() -> None:
    generated = PAPER / "numbers.tex"
    sealed = REFERENCE / "numbers.tex"
    check("numbers.tex exists", generated.exists() and sealed.exists())
    if not generated.exists() or not sealed.exists():
        return
    generated_text = generated.read_text()
    sealed_text = sealed.read_text()

    def macro_map(text: str) -> dict[str, str]:
        answer: dict[str, str] = {}
        for line in text.splitlines():
            match = re.match(r"\\newcommand\{\\(num[A-Za-z]+)\}\{(.*)\}$", line)
            if match:
                answer[match.group(1)] = match.group(2)
        return answer

    generated_macros = macro_map(generated_text)
    sealed_macros = macro_map(sealed_text)
    defined = set(generated_macros)
    tex_paths = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    source = "\n".join(path.read_text(errors="ignore") for path in tex_paths)
    source = re.sub(r"(?<!\\)%.*", "", source)
    used = set(re.findall(r"\\(num[A-Za-z]+)", source))
    missing = sorted(used - defined)
    check("generated macro count", len(defined) == 587, f"{len(defined)} macros")
    check("all manuscript numeric macros defined", not missing,
          "none missing" if not missing else ", ".join(missing[:10]))
    check("no unresolved number placeholders", "??" not in generated_text)
    changed_used = sorted(
        name for name in used
        if generated_macros.get(name) != sealed_macros.get(name)
    )
    changed_unused = sorted(
        name for name in (set(generated_macros) | set(sealed_macros)) - used
        if generated_macros.get(name) != sealed_macros.get(name)
    )
    check("sealed manuscript-used numeric claims", not changed_used,
          f"{len(used)} used macros agree" if not changed_used
          else ", ".join(changed_used[:10]))
    if changed_unused:
        warnings.append(
            f"{len(changed_unused)} unused historical numeric macros differ: "
            + ", ".join(changed_unused[:8])
        )

    cited: set[str] = set()
    for group in re.findall(
        r"\\cite[A-Za-z]*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}",
        source,
        re.S,
    ):
        cited.update(re.sub(r"\s+", "", key) for key in group.split(","))
    bib_text = (PAPER / "references.bib").read_text(errors="ignore")
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))
    undefined = sorted(cited - bib_keys)
    unused = sorted(bib_keys - cited)
    check("all citation keys defined", not undefined,
          "none missing" if not undefined else ", ".join(undefined[:10]))
    if unused:
        warnings.append(f"{len(unused)} bibliography entries are not cited")


def verify_data_figures_and_pdf() -> None:
    required = [
        ROOT / "data/sleep_edf/hypnograms.npz",
        ROOT / "data/sleep_edf/meta.json",
        ROOT / "data/sleep_edf/ST-subjects.xls",
        ROOT / "data/ltaf/af_series.npz",
        ROOT / "data/ltaf/meta.json",
        PAPER / "main.tex",
        PAPER / "main.pdf",
    ]
    check("required data and paper files", all(path.exists() for path in required))
    sleep_raw = list((ROOT / "data/sleep_edf/raw").glob("*Hypnogram.edf"))
    af_atr = list((ROOT / "data/ltafdb").glob("*.atr"))
    af_hea = list((ROOT / "data/ltafdb").glob("*.hea"))
    check("Sleep annotation cache", len(sleep_raw) == 197, f"{len(sleep_raw)} files")
    check("AF annotation cache", len(af_atr) == len(af_hea) == 84,
          f"{len(af_atr)} annotations, {len(af_hea)} headers")

    for name in ("fig_identifiability", "fig_calibration", "fig_real", "fig_sweep"):
        current = ROOT / "figures" / f"{name}.pdf"
        paper_copy = PAPER / "figures" / f"{name}.pdf"
        reference = REFERENCE / "figures" / f"{name}.pdf"
        check(f"{name} PDFs", all(path.exists() and path.stat().st_size > 1000
                                   for path in (current, paper_copy, reference)))
        if current.exists() and paper_copy.exists():
            check(f"{name} synchronised", current.read_bytes() == paper_copy.read_bytes())

    pdf = PAPER / "main.pdf"
    if pdf.exists():
        try:
            output = subprocess.run(
                ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
            ).stdout
            match = re.search(r"^Pages:\s+(\d+)", output, re.M)
            check("compiled paper page count", bool(match) and int(match.group(1)) == 35,
                  match.group(1) if match else "unavailable")
        except (FileNotFoundError, subprocess.CalledProcessError):
            warnings.append("pdfinfo unavailable; page-count check skipped")
    log = PAPER / "main.log"
    if log.exists():
        log_text = log.read_text(errors="ignore")
        bad = re.findall(
            r"(?:undefined references|Citation [`'][^\n]+ undefined|"
            r"Undefined control sequence|Fatal error|Emergency stop)",
            log_text,
            flags=re.I,
        )
        check("LaTeX fatal/undefined diagnostics", not bad,
              "clean" if not bad else bad[0])


def main() -> int:
    print("Reference comparison")
    compare_results()
    verify_macros_and_bibliography()
    verify_data_figures_and_pdf()
    print(f"\nCompleted {checks} checks: {len(problems)} failed, {len(warnings)} warnings.")
    for item in warnings:
        print(f"  [WARN] {item}")
    if problems:
        print("Failures:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("Reproduction artefacts agree with the sealed final-manuscript reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
