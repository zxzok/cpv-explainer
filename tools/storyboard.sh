#!/usr/bin/env bash
# Render the storyboard (final frame of every beat) to PNG files with headless Chrome.
#   tools/storyboard.sh [out_dir] [lang] [base_url] [chapters...]
#   e.g. tools/storyboard.sh shots en http://127.0.0.1:8791/explainer/ e0 e1 e2 e3 e4 e5 e6
# Requires the site to be served (python3 -m http.server 8791 --directory site).
set -uo pipefail
OUT="${1:-storyboard}"; LANG_="${2:-en}"; BASE="${3:-http://127.0.0.1:8791/technical/}"; shift 3 2>/dev/null || true
CHAPTERS=("$@"); [ ${#CHAPTERS[@]} -eq 0 ] && CHAPTERS=(ch0 ch1 ch2 ch3 ch4 ch5 ch6 ch7 ch8 ch9)
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p "$OUT"
for ch in "${CHAPTERS[@]}"; do
  PROFILE="$(mktemp -d)"; t0=$(date +%s)
  timeout 50 "$CH" --headless=new --hide-scrollbars --window-size=1500,2600 --virtual-time-budget=12000 \
        --user-data-dir="$PROFILE" --no-first-run --disable-extensions --screenshot="$OUT/$ch-$LANG_.png" \
        "${BASE}?sheet=$ch&cols=2&lang=$LANG_" >/dev/null 2>&1 || echo "failed/timeout: $ch"
  rm -rf "$PROFILE"
  echo "$OUT/$ch-$LANG_.png ($(( $(date +%s) - t0 )) s)"
done
