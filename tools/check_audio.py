#!/usr/bin/env python3
"""Listen back to every narration clip with SenseVoice and compare it to the script.

Synthesised speech occasionally swallows a first word, stutters, or garbles a
number; this catches it without anyone sitting through ten minutes of audio.
For each clip it prints the duration and a word-level match score (1.0 = the
transcript reproduces the script word for word, after normalisation), and
lists clips below the threshold so they can be regenerated:

    python3 tools/check_audio.py                       # both languages
    python3 tools/check_audio.py --lang en --min 0.85
    python3 tools/check_audio.py --only ch4-b1,ch7-b2  # re-check a few
    python3 tools/check_audio.py --regen               # delete flagged clips and regenerate them

The ASR runs without inverse text normalisation so that numbers are compared in
their spoken form ("zero point six eight"), matching how the script is written.
Must run with the VoxCPM venv python (funasr lives there); the script re-execs
itself if started with another interpreter.
"""
import argparse, difflib, json, os, pathlib, re, subprocess, sys

VENV_PY = os.path.expanduser("~/语音合成/VoxCPM/.venv/bin/python")
if sys.executable != VENV_PY and os.path.exists(VENV_PY):
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.execv(VENV_PY, [VENV_PY] + sys.argv)

SITE = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["technical", "explainer"], default="technical")
ap.add_argument("--lang", choices=["en", "zh", "both"], default="both")
ap.add_argument("--min", type=float, default=0.88, help="flag clips whose word match is below this")
ap.add_argument("--only", default="", help="comma-separated clip ids (ch4-b1,...) to check")
ap.add_argument("--regen", action="store_true", help="delete flagged clips and regenerate them (one pass)")
args = ap.parse_args()
SCRIPT = SITE / f"narration/{args.set}.json"; AUDIO = SITE / "audio" / args.set

sys.path.insert(0, os.path.expanduser("~/.claude/skills/voxcpm-tts"))
import helper  # noqa: E402

def transcribe(path):
    res = helper.get_asr().generate(input=str(path), language="auto", use_itn=False)
    return res[0]["text"].split("|>")[-1].strip()

EN_MAP = {"r squared": "r squared", "minus": "minus", "optimisation": "optimization", "optimise": "optimize",
          "standardised": "standardized", "standardise": "standardize", "realised": "realized", "generalised": "generalized",
          "centred": "centered", "modelling": "modeling", "recognised": "recognized", "summarising": "summarizing"}
_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
def words_to_digits(tokens):
    """Collapse spoken numbers ("one hundred and ninety seven", "zero point six eight") into digit strings, so the
    script can be compared with an ASR transcript that prints digits."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _UNITS or t == "hundred":
            val, j, seen = 0, i, False
            while j < len(tokens):
                w = tokens[j]
                if w in _UNITS: val += _UNITS[w]; seen = True
                elif w == "hundred" and seen: val *= 100
                elif w == "thousand" and seen: val *= 1000
                elif w == "and" and seen and j + 1 < len(tokens) and tokens[j + 1] in _UNITS: pass
                else: break
                j += 1
            if seen and tokens[j - 1] != "and":
                digits = str(val)
                if j < len(tokens) and tokens[j] == "point":          # decimals: zero point six eight -> 0.68
                    k, frac = j + 1, ""
                    while k < len(tokens) and tokens[k] in _UNITS and _UNITS[tokens[k]] < 10: frac += str(_UNITS[tokens[k]]); k += 1
                    if frac: digits += "." + frac; j = k
                out.append(digits); i = j; continue
        out.append(t); i += 1
    return out
def norm_en(s):
    s = s.lower().replace("—", " ").replace("–", " ").replace("-", " ").replace("’", "'")
    s = re.sub(r"[^a-z0-9'. ]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)          # keep the dot only inside decimals
    s = " ".join(s.split())
    for k, v in EN_MAP.items():
        s = s.replace(k, v)
    toks = words_to_digits(s.split())
    # ASR glues digits to neighbours ("minus004", "measurements120"): split them back apart
    return re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", " ".join(toks)).split()
def norm_zh(s):
    s = s.lower()
    return list(re.sub(r"[^一-鿿a-z0-9]", "", s))

def score(want, heard, lang):
    a, b = (norm_zh(want), norm_zh(heard)) if lang == "zh" else (norm_en(want), norm_en(heard))
    if not a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def duration(p):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0

data = json.loads(SCRIPT.read_text())
langs = ["en", "zh"] if args.lang == "both" else [args.lang]
only = {x.strip() for x in args.only.split(",") if x.strip()}
flagged = []
for lang in langs:
    print(f"== {lang}")
    for ch in data["chapters"]:
        for b in ch["beats"]:
            cid = f"{ch['id']}-{b['id']}"
            if only and cid not in only:
                continue
            p = AUDIO / lang / f"{cid}.m4a"
            if not p.exists():
                print(f"  MISSING {p.name}"); flagged.append((lang, p)); continue
            heard = transcribe(p)
            want = b.get(lang + "_tts") or b[lang]
            sim = score(want, heard, lang)
            mark = "" if sim >= args.min else "   <-- check"
            print(f"  {p.name:12s} {duration(p):5.1f}s  match {sim:.2f}{mark}")
            if sim < args.min:
                flagged.append((lang, p)); print(f"      heard: {heard}")
print()
if not flagged:
    print("all clips match the script"); sys.exit(0)
print("flagged:", ", ".join(f"{l}/{p.name}" for l, p in flagged))
if args.regen:
    for _, p in flagged:
        p.unlink(missing_ok=True)
    langs_flagged = sorted({l for l, _ in flagged})
    for l in langs_flagged:
        subprocess.run(["python3", str(SITE / "tools/make_audio.py"), "--set", args.set, "--lang", l], check=True)   # system python; make_audio spawns the venv itself
    print("regenerated; run the check again to confirm")
else:
    print("regenerate with: python3 tools/check_audio.py --regen  (or delete the files and run make_audio.py)")
