# Counterfactual Evaluation of Temporal Observation Protocols

**Xizhe Zhang (张锡哲)** · School of Biomedical Engineering and Informatics, Nanjing Medical University

This repository holds everything released with the paper:

| What | Where | Notes |
|---|---|---|
| **Explainer website** (live at **<https://cpv.xizhe.net>**) | repository root: `index.html`, `explainer/`, `technical/`, `js/`, `css/`, `narration/`, `audio/` | three tiers — a landing page, a four-minute narrated story, a ten-minute interactive technical tour; English and Chinese |
| **Manuscript (PDF)** | [`paper/main.pdf`](paper/main.pdf) | the submitted version |
| **Reproduction code & data package** | [`code/`](code/) — start with [`code/README.md`](code/README.md) | methods library, every simulation and real-data experiment, unit tests, LaTeX sources, cached open annotation data, archived results and reference figures; `make verify-quick` / `make all` |
| Key figures, social card, 30-second clip | [`figures/`](figures/), [`assets/`](assets/) | PNG / MP4 / GIF for talks and posts |
| Full narration script (both languages) | [`CONTENT.md`](CONTENT.md) | generated from `narration/*.json` |

## The paper in one paragraph

A benchmark collected under one observation protocol (when, how often, how long and how precisely a latent trajectory
is measured) does not in general determine how well a *different, never-deployed* protocol would predict a
trajectory-level target — even with infinitely many subjects. The paper gives the exact condition under which the
value of an alternative protocol is identified, a four-point counterexample where two latent worlds agree on every
benchmark quantity yet assign the alternative protocol values 0.682 and 0.827, the calibration data that restore
identification (targeted augmentation or a small densely observed subset), uniform error bounds that turn into a
decision guarantee (regret ≤ 2ε), exact rank-one marginal gains for target-aware protocol design, and real-data
analyses on Sleep-EDF Expanded and the Long-Term AF Database.

---

## 1. The website (`/`)

Plain HTML/CSS/JS — no framework, no build step. Open `index.html` through any static server (or see "Local preview").

```
index.html              landing page (EN/ZH pairs as .en/.zh elements), interactive figures, citation, metadata
explainer/index.html    four-minute narrated story (7 chapters, 21 beats)
technical/index.html    ten-minute technical tour (10 chapters, 43 beats)
figures/index.html      renders the downloadable key figures
assets/social.html      the 1200×630 social card

css/landing.css         landing page styles          css/site.css   player ("theatre") pages
js/config.js            site URL, paper/code links, BibTeX — edit this first when moving the site
js/data.js              every number shown or spoken, extracted from paper/numbers.tex and results/ (tools/extract_data.py)
js/math.js              the paper's formulas in JavaScript (protocol value I_g, Q_S, greedy design, ...)
js/mathtext.js          TeX-subset typesetter used for every formula, on canvas and in prose
js/engine.js            Stage (tweens, WebGL ribbon field, storyboard mode) and Player (narration, timeline, seeking)
js/gl.js                WebGL latent-trajectory field          js/sheet.js   storyboard (?sheet=) mode
js/landing.js           landing-page interactives (two worlds, cohort, resolution, draggable design demo)
js/scenes/ch0…ch9.js    the ten technical chapters (each exports beats that drive the stage)
js/scenes_explainer.js  the four-minute story, composed from the technical scenes
narration/*.json        SOURCE OF TRUTH for all spoken/read text, per beat, EN + ZH
js/narration-*.js       generated from the JSON + clip durations (tools/build_narration.py)
audio/<set>/<lang>/     narration clips (m4a), one per beat
```

### Editing the narration or the pages

```bash
# 1. edit narration/technical.json or narration/explainer.json (text, zh_tts/en_tts overrides, read panels)
python3 tools/make_audio.py --set technical            # regenerate missing clips (VoxCPM2 voice clone; --force for all)
python3 tools/check_audio.py --set technical --regen   # ASR round-trip check; re-record clips that drifted
python3 tools/build_narration.py --set technical       # write js/narration-technical.js with clip durations
python3 tools/export_content.py                        # refresh CONTENT.md
python3 tools/build_single.py --page technical --lang en,zh   # optional: single-file offline bundle → dist/
```

`tools/extract_data.py` rebuilds `js/data.js` from the paper's `numbers.tex` and results; `tools/render.mjs`
re-renders the figures, social card and the 30-second clip with headless Chrome; `tools/storyboard.sh` captures
every beat of a chapter as a contact sheet for review.

### Local preview

```bash
python3 tools/serve.py 8791          # http://127.0.0.1:8791 — supports HTTP Range, so seeking inside clips works
```

### Deployment

GitHub Pages serves the `main` branch root at <https://cpv.xizhe.net> (`CNAME`). Push to `main` to deploy.

---

## 2. The reproduction package (`code/`)

`code/` is the complete package released with the final manuscript. Read [`code/README.md`](code/README.md)
for the full instructions; in short:

```bash
cd code
make setup            # Python 3.11–3.14 virtual environment + dependencies
make verify-quick     # unit tests, regenerate figures + numbers.tex from archived results, compile the paper, compare with reference/
make all              # full re-run from the cached data (≈ 60–90 min on a 24-core Apple Silicon machine; do not use -j)
```

```
code/
├── README.md, REPRODUCIBILITY_REPORT.md, DATA_PROVENANCE.md, LICENSE-NOTICE.md, CITATION.cff
├── Makefile, pyproject.toml, requirements.txt, config/
├── protocol_ceiling/      methods library: Gaussian protocol value, invisible directions, calibration estimator,
│                          rank-one design, permutation and augmentation constructions
├── experiments/           every simulation (S1–S9) and real-data experiment, cross-fitting, resampling, sensitivity
├── tests/                 unit tests (run by make verify-quick)
├── data/                  cached PhysioNet annotation files and processed arrays (Sleep-EDF Expanded, Long-Term AF);
│                          no raw PSG/ECG waveforms; experiments/fetch_data.py re-downloads from pinned paths
├── results/               archived outputs of the final runs           reference/   sealed outputs for comparison
├── figures/               paper figures as produced by the scripts
├── paper/                 LaTeX sources of the final manuscript; numbers.tex holds the 587 numeric macros
├── scripts/               helpers (numbers.tex generation, figure sync)
├── validation/            logs of the final `make verify-quick` and full reproduction runs
└── SHA256SUMS             checksums of every file in the package (`cd code && shasum -a 256 -c SHA256SUMS`)
```

Every number on the website is read from `code/paper/numbers.tex` and `code/results/`; the interactive scenes
recompute protocol values in the browser with the same formulas (`js/math.js`).

---

## Citation

```bibtex
@unpublished{zhang2026counterfactual,
  title  = {Counterfactual Evaluation of Temporal Observation Protocols},
  author = {Zhang, Xizhe},
  year   = {2026},
  note   = {Manuscript. Explainer: https://cpv.xizhe.net}
}
```

Data: Sleep-EDF Expanded and the Long-Term AF Database (PhysioNet, open access); only expert annotation files are
used — see `code/DATA_PROVENANCE.md` and `code/LICENSE-NOTICE.md`.

---

## 中文说明

- 网站源码在仓库根目录（`index.html` 首页、`explainer/` 四分钟导览、`technical/` 技术导览、`js/` `css/` `narration/` `audio/`），线上地址 <https://cpv.xizhe.net>。
- 论文 PDF 在 `paper/main.pdf`。
- **论文的复现代码与数据包在 `code/`**（方法库 `protocol_ceiling/`、全部实验 `experiments/`、单元测试、论文 LaTeX 源文件、缓存的 PhysioNet 标注数据、归档结果与参考图）；用法见 `code/README.md`：`make setup && make verify-quick`（快速验证）或 `make all`（完整重跑）。
- 解说文本以 `narration/*.json` 为准，改完后依次运行 `tools/make_audio.py`、`tools/check_audio.py`、`tools/build_narration.py`；推送到 `main` 即自动部署。
