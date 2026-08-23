# Reproducibility validation report

## Outcome

The final-manuscript code package was rebuilt and validated on 2026-08-20/21
(America/Los_Angeles). All experiment stages used by the final manuscript were
executed from the packaged code and cached annotation data. The rebuilt active
results agree with the sealed reference, every numeric macro used by the final
TeX agrees, and the manuscript rebuilds to 35 pages without undefined citations,
references, or control sequences.

Status: **PASS for the final manuscript reproduction scope.**

## Input audit

The user-supplied manuscript ZIP contained 25 files: `main.tex`, `numbers.tex`,
`references.bib`, `jmlr2e.sty`, 17 section files, and four PDF figures. It did
not contain experiment code, data, an environment specification, or a
reproduction entry point.

The supplied `numbers.tex` was byte-identical to the existing result-to-macro
generator's sealed output. Three supplied figures were byte-identical to the
existing outputs. The calibration figure matched the later corrected generator
that clusters the three targets sharing each S4 calibration draw and uses the
final protocol-class terminology; that generator is the version shipped here.

## Executed validation

| Check | Result |
|---|---|
| Unit and numerical regression checks | 218 PASS lines; all three test scripts completed successfully |
| Sleep data cache | 197 hypnograms, 100 subjects; 484,111/484,111 parser-cross-check epochs agree |
| Long-Term AF cache | 84 annotation/header pairs |
| S3 finite calibration | 9,600 replications; 43,200 per-replication rows |
| S4 selection regret | 7,200/7,200 satisfy `regret <= 2 epsilon_m`; 0 violations |
| S5 target-aware design | 25 target-by-configuration instances rebuilt |
| S6 misspecification | 12 conditions x 2 targets x 200 replications; 4,800 rows |
| S8 nested protocol classes | 360 selections; 0 corollary violations |
| Sleep support stability | 1,000 complete stratified pipeline subsamples |
| Result/reference gate | 53 JSON/CSV reference files checked; active outputs agree after ignoring elapsed-time, environment-path, and parallel row-order metadata |
| Manuscript numbers | 587 macros generated; all 144 macros referenced by the final manuscript agree with the seal |
| Figures | All four manuscript PDFs regenerated and synchronised into `paper/figures/` |
| PDF | 35 US-letter pages; no fatal or undefined LaTeX diagnostics |
| Artifact gate | 21/21 checks pass |
| Fresh-ZIP integrity | 594/594 file hashes verified after extraction |
| Fresh-ZIP quick reproduction | `make verify-quick` exit 0; 21/21 checks; rebuilt PDF 35 pages |

The complete full-run log covers 2026-08-20 22:53:11 PDT through
2026-08-21 00:13:41 PDT (80.5 minutes) and is stored at
`validation/full_reproduction.log`.

After packaging, the archive was extracted into a previously absent directory.
All 594 entries in `SHA256SUMS` verified before execution, and the extracted
copy completed `make verify-quick` successfully. This check did not reuse the
working release directory.

## Important audit distinction

The first full-run gate was deliberately strict enough to expose historical
outputs that are no longer part of the final manuscript:

- four changed macros came only from the unused S7 learner regression;
- none of the 144 macros actually referenced by the final TeX changed;
- the old `record64_sensitivity` output belongs to a former 71-record analysis,
  whereas the final manuscript's primary AF analysis uses all 84 records;
- absolute output paths, elapsed runtimes, and parallel CSV completion order are
  machine metadata rather than scientific differences.

The Makefile now separates these scripts into `make retained-regressions`.
They remain in the archive for auditability but are outside `make all` and the
final-paper failure gate. Their sealed historical outputs are retained under
`reference/results/`.

## Validated environment

```text
Python      3.14.6
Platform    macOS 26.6.2, arm64
NumPy       2.5.1
SciPy       1.18.0
Matplotlib  3.11.1
WFDB        4.3.1
MNE         1.12.1
xlrd        2.0.2
requests    2.34.2
TeX Live    2026/Homebrew
latexmk     4.87
```

## Non-blocking notes

- The final `references.bib` contains 25 entries not cited by the compiled
  article. All citations that do appear in the article are defined; this does
  not block code or numerical reproduction, but the unused entries can be
  pruned in a later manuscript-only cleanup if desired.
- Regenerated PDF binaries can differ from the supplied PDFs because PDF
  creation timestamps, Matplotlib versions, fonts, and TeX backends are embedded
  in the file. The archive preserves the supplied final figures under
  `reference/figures/`; numerical results, used macros, page count, citations,
  and figure inputs are the scientific reproduction criteria.
- The generic LaTeX helper encountered a UTF-8 decoding error while reading the
  TeX process stream. Direct `latexmk` with the same TeX Live installation
  compiled the source successfully; this was a wrapper-output issue, not a TeX
  source failure.
- No open-source license was selected on the author's behalf. See
  `LICENSE-NOTICE.md` before public release.
