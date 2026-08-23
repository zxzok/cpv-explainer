# Explainer website — maintenance notes

The interactive explainer for the paper lives in this repository's root and is served at <https://cpv.xizhe.net>.
This file documents how it is built and edited; the paper and the reproduction code are described in [README.md](README.md).

## Layout

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


