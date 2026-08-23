# Data provenance

The real-data experiments use only open annotation files; raw PSG and ECG
waveforms are neither needed nor included.

## Sleep-EDF Expanded

- Fixed source: `https://physionet.org/files/sleep-edfx/1.0.0/`
- Included material: 197 `*-Hypnogram.edf` annotation files and
  `ST-subjects.xls`.
- Processed output: `data/sleep_edf/hypnograms.npz` and `meta.json`.
- Analysis units: 197 recordings from 100 subjects. Subject identifiers are
  used to keep repeated nights within the same fold.

## Long-Term AF Database

- Fixed source: `https://physionet.org/files/ltafdb/1.0.0/`
- Included material: rhythm annotation (`.atr`) and header (`.hea`) files for
  84 records.
- Processed output: `data/ltaf/af_series.npz` and `meta.json`.
- Analysis begins at the first rhythm marker; an unannotated prefix is not
  coded as non-AF.

`experiments/fetch_data.py` implements downloading, parsing and provenance
checks. It caches plausible existing files, compares MNE and a self-contained
EDF+ TAL parser for Sleep annotations, and writes `results/data_summary.*`.
Use the source pages above for the current PhysioNet terms and required dataset
citations. File hashes for this sealed archive are listed in `SHA256SUMS`.

