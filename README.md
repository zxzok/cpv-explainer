# Counterfactual Evaluation of Temporal Observation Protocols — explainer site

Three tiers: `/` (landing page), `/explainer/` (four-minute narrated story), `/technical/` (ten-minute interactive tour).
Plain HTML/CSS/JS, no build step. Narration clips live in `audio/`; `narration/*.json` is the source of truth
(`python3 tools/build_narration.py --set technical|explainer` regenerates `js/narration-*.js`).
Local preview with HTTP Range support: `python3 tools/serve.py 8791`.

Xizhe Zhang (张锡哲) · Nanjing Medical University
