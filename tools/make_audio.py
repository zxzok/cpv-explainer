#!/usr/bin/env python3
"""Generate the narration clips from narration/<set>.json (technical or explainer).

The engine and voices default to `meta.engine` / `meta.voices` in script.json
(currently VoxCPM2 with the cloned voice 我的声音); `--engine say` falls back to
macOS voices (Samantha / Tingting).  Clips are encoded to AAC with ffmpeg.  Each clip is written to audio/<lang>/<chapter>-<beat>.m4a
and skipped when it already exists, so re-running after editing one sentence
only regenerates that sentence (delete the clip to force it).

    python3 tools/make_audio.py                 # both languages, say
    python3 tools/make_audio.py --lang en       # one language
    python3 tools/make_audio.py --force         # regenerate everything
    python3 tools/make_audio.py --engine voxcpm --voice-en Narrator --voice-zh 旁白

The VoxCPM engine calls the local helper in ~/.claude/skills/voxcpm-tts (see its
SKILL.md); any other TTS can be plugged in by writing WAV/M4A files with the
same names.  After generating audio, run tools/build_narration.py so the player
knows the clip durations.
"""
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile

SITE = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = None  # set below from --set
AUDIO = None

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["technical", "explainer"], default="technical", help="which narration set: narration/<set>.json -> audio/<set>/")
ap.add_argument("--lang", choices=["en", "zh", "both"], default="both")
ap.add_argument("--engine", choices=["say", "voxcpm"], default=None, help="default: meta.engine in script.json, else say")
ap.add_argument("--voice-en", default=None)
ap.add_argument("--voice-zh", default=None)
ap.add_argument("--rate-en", type=int, default=168, help="say words per minute (English)")
ap.add_argument("--rate-zh", type=int, default=180, help="say rate (Chinese)")
ap.add_argument("--bitrate", default="64k")
ap.add_argument("--steps", type=int, default=14, help="VoxCPM inference steps (quality vs time)")
ap.add_argument("--target-en", type=float, default=2.55, help="target English rate, words per second (about 150 wpm)")
ap.add_argument("--target-zh", type=float, default=4.4, help="target Chinese rate, characters per second")
ap.add_argument("--no-rate", action="store_true", help="do not normalise the speaking rate with atempo")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
SCRIPT = SITE / f"narration/{args.set}.json"; AUDIO = SITE / "audio" / args.set

data = json.loads(SCRIPT.read_text())
if args.engine is None:
    args.engine = data["meta"].get("engine", "say")
voices = {"en": args.voice_en or data["meta"]["voices"]["en"],
          "zh": args.voice_zh or data["meta"]["voices"]["zh"]}
langs = ["en", "zh"] if args.lang == "both" else [args.lang]

if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg not found (brew install ffmpeg)")

def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0

def speech_units(text, lang):
    import re
    return len(text.split()) if lang == "en" else len(re.sub(r"[^一-鿿]", "", text))

def encode(src, dst, text=None, lang=None):
    """Trim silence, optionally bring the speaking rate towards the target (pitch-preserving atempo,
    at most ±12 %, only when the clip is more than 5 % off), pad a short breath, encode to AAC."""
    trim = ("silenceremove=start_periods=1:start_threshold=-50dB,"
            "areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse")
    tempo = ""
    if text and lang and not args.no_rate:
        with tempfile.TemporaryDirectory() as td:
            probe = pathlib.Path(td) / "trim.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af", trim, str(probe)], check=True)
            dur = duration(probe)
        rate = speech_units(text, lang) / max(dur, 0.1)
        target = args.target_en if lang == "en" else args.target_zh
        f = max(0.8, min(1.15, target / rate))   # atempo>1 speeds up, so a fast clip (rate>target) gets f<1; pitch-preserving
        if abs(f - 1) > 0.05:
            tempo = f",atempo={f:.3f}"
            print(f"    rate {rate:.2f}/s -> atempo {f:.3f}", file=sys.stderr)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
                    "-af", trim + tempo + ",apad=pad_dur=0.45",
                    "-c:a", "aac", "-b:a", args.bitrate, "-ar", "44100", str(dst)], check=True)

def say_clip(text, lang, dst):
    rate = args.rate_en if lang == "en" else args.rate_zh
    with tempfile.TemporaryDirectory() as td:
        aiff = pathlib.Path(td) / "clip.aiff"
        subprocess.run(["say", "-v", voices[lang], "-r", str(rate), "-o", str(aiff), text], check=True)
        encode(aiff, dst, text, lang)

def voxcpm_batch(items, lang):
    """items: list of (text, dst).  One model load for the whole language."""
    py = os.path.expanduser("~/语音合成/VoxCPM/.venv/bin/python")
    helper = os.path.expanduser("~/.claude/skills/voxcpm-tts/helper.py")
    with tempfile.TemporaryDirectory() as td:
        script = pathlib.Path(td) / "lines.txt"
        script.write_text("\n".join(t.replace("\n", " ") for t, _ in items))
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", TOKENIZERS_PARALLELISM="false")
        res = subprocess.run([py, helper, "batch", "--voice", voices[lang], "--script", str(script),
                              "--outdir", td, "--steps", str(args.steps)], env=env, capture_output=True, text=True)
        sys.stderr.write(res.stderr[-3000:])
        line = [l for l in res.stdout.splitlines() if l.startswith("@@VOXCPM_RESULT@@")]
        if not line:
            sys.exit("voxcpm failed:\n" + res.stderr[-2000:])
        out = json.loads(line[-1][len("@@VOXCPM_RESULT@@"):])
        paths = [c["output"] for c in out.get("clips", [])]
        if len(paths) != len(items):
            sys.exit(f"voxcpm returned {len(paths)} clips for {len(items)} lines: {out}")
        for (text, dst), p in zip(items, paths):
            encode(p, dst, text, lang)

for lang in langs:
    outdir = AUDIO / lang
    outdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for ch in data["chapters"]:
        for b in ch["beats"]:
            dst = outdir / f"{ch['id']}-{b['id']}.m4a"
            if dst.exists() and not args.force:
                continue
            text = b.get(lang + "_tts") or b[lang]
            todo.append((text, dst))
    print(f"[{lang}] {len(todo)} clips to generate with {args.engine} ({voices[lang]})")
    if args.engine == "say":
        for i, (text, dst) in enumerate(todo, 1):
            say_clip(text, lang, dst)
            print(f"  {i:3d}/{len(todo)} {dst.name}")
    elif todo:
        voxcpm_batch(todo, lang)
print(f"done; now run: python3 tools/build_narration.py --set {args.set}")
