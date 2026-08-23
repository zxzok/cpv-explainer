#!/usr/bin/env python3
"""Export the site's content as one Markdown document for editing.

Pulls everything a writer might change from the real sources — the narration
and Read text from narration/script.json, the on-screen captions from the scene
files, the chapter durations from js/narration.js — so the document never
drifts from the site.  Re-run after editing; or edit script.json directly.

    python3 tools/export_content.py            # writes CONTENT.md next to index.html
"""
import html, json, pathlib, re

SITE = pathlib.Path(__file__).resolve().parents[1]
SETS = {}
for name in ("technical", "explainer"):
    script_ = json.loads((SITE / f"narration/{name}.json").read_text())
    nj = (SITE / f"js/narration-{name}.js").read_text()
    gen = json.loads(nj[nj.index("{"):nj.rindex("}") + 1])
    SETS[name] = (script_, {(c["id"], b["id"]): b.get("dur", {}) for c in gen["chapters"] for b in c["beats"]})
script, durs = SETS["technical"]

SCENE_FILES = {"ch0": "ch0_hero.js", "ch1": "ch1_question.js", "ch2": "ch2_value.js", "ch3": "ch3_twoworlds.js",
               "ch4": "ch4_invisible.js", "ch5": "ch5_restore.js", "ch6": "ch6_resolution.js", "ch7": "ch7_design.js",
               "ch8": "ch8_real.js", "ch9": "ch9_synthesis.js"}
PAPER = {"ch0": "Abstract / §1", "ch1": "§1", "ch2": "§2 · Definition 1, Proposition 2", "ch3": "§3 · Theorem 3, Figure 1",
         "ch4": "§3 · Definition 4, Theorem 5, Proposition 6, Example 7", "ch5": "§3.5–4 · Proposition 8, Theorems 10–11, Figure 2a",
         "ch6": "§4 · Corollary 12, Figure 2b–d", "ch7": "§5 · Proposition 13, Lemma 14, Table 1", "ch8": "§6.4 · Figures 3–4", "ch9": "§8"}
VISUAL = {
    "ch0": ("WebGL：260 条潜在轨迹的发光带；A 的蓝色观测线、B 的品红虚线；片头标题卡", "WebGL field of 260 latent trajectories; blue ticks for protocol A, dashed magenta for B; title card"),
    "ch1": ("整夜睡眠分期图（toy hypnogram）；REM 比例读数；A 的单个片段、B 的四个片段；n → ∞ 与“No.”；两个时刻之间的弧线", "Toy hypnogram; REM-fraction read-out; A's single epoch vs B's four; the n → ∞ chain and “No.”; the arc between two times"),
    "ch2": ("定义 1 的公式与价值表；K 与 Q_S 热力图、ω 权重条、A_S 行；可拖动的三个测量时刻，均值/占用时间两个目标实时计算", "Definition 1 with a gauge; K and Q_S heat maps, ω strip, A_S rows; three draggable measurement times with live mean / occupation values"),
    "ch3": ("四点网格；相关函数 ρ₀、ρ₊、ρ₋ 折线；三个可观测量的两列数字与最大差异；I(B;K±) 两个价值表；可拖动的 ε 滑块；Θ–Z₁–Z₂ 重叠示意", "Four-point grid; ρ₀/ρ₊/ρ₋ profiles; the three observables in both worlds and their discrepancy; gauges for I(B;K±); draggable ε slider; the Θ–Z₁–Z₂ overlap diagram"),
    "ch4": ("16×16 热力图只亮 4×4 子块；三块公式与不可见条件；120 vs 15 的计数；定理 5 的导数；Z₁↔Z₂ 置换动画与 K_a / K_a′ 热力图", "16×16 heat map with the 4×4 visible block lit; the three blocks and the invisibility conditions; the 120 vs 15 count; Theorem 5's derivative; the Z₁↔Z₂ swap with K_a / K_a′"),
    "ch5": ("四点补测与不可见维数 4 → 2 → 0；60 条密集校准轨迹（WebGL）；四步估计流水线；定理 10 的 β 表；图 2a 的对数–对数误差曲线（真实结果）", "Augmentation with invisible dimension 4 → 2 → 0; 60 dense calibration paths (WebGL); the four-step estimator; Theorem 10's β table; Figure 2a log–log error curves from the results files"),
    "ch6": ("“≤ 2ε”的三行论证与误差条；四层嵌套候选类与 S*；图 2c,d 的损失与一致误差曲线；结论卡", "The three-line “≤ 2ε” argument with error brackets; four nested candidate classes and S*; Figure 2c,d regret and uniform-error curves; takeaway card"),
    "ch7": ("后验轨迹带（WebGL）随贪心加点塌缩；候选目录与动作五元组；秩一收益公式；各候选的边际收益条；两种目标选出的时刻；表 1 的效率条", "Posterior band (WebGL) collapsing as greedy adds measurements; the catalogue and the action tuple; rank-one gain formula; per-candidate gain bars; times chosen by two targets; Table 1 efficiency bars"),
    "ch8": ("两张数据卡（Sleep-EDF / LTAF）；REM 分散 vs 连续柱图；房颤负担折线；图 4 的支持稳定性曲线与重抽样区间", "Two dataset cards; REM dispersed-vs-contiguous bars; AF-burden lines; Figure 4 support-stability curves and resampling ranges"),
    "ch9": ("识别 / 校准 / 设计三列；一句话原则；发布信息卡", "Three columns identification / calibration / design; the one-sentence principle; release card"),
}

def html_to_md(s):
    s = re.sub(r"<span class='f'>(.*?)</span>", r"`\1`", s)
    s = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", s)
    s = re.sub(r"<(em|i)>(.*?)</\1>", r"*\2*", s)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s)
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
    s = re.sub(r"</?p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()

def screen_text(ch_id):
    src = (SITE / "js/scenes" / SCENE_FILES[ch_id]).read_text()
    pairs, seen = [], set()
    for m in re.finditer(r'S\.t\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"', src):
        en, zh = m.group(1).replace('\\"', '"'), m.group(2).replace('\\"', '"')
        if (en, zh) not in seen:
            seen.add((en, zh)); pairs.append((en, zh))
    for m in re.finditer(r'S\.header\(ctx,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"', src):
        en, zh = m.group(1), m.group(2)
        if (en, zh) not in seen:
            seen.add((en, zh)); pairs.insert(0, (en, zh))
    for m in re.finditer(r'sub:\s*\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]', src):
        en, zh = m.group(1), m.group(2)
        if (en, zh) not in seen:
            seen.add((en, zh)); pairs.append((en, zh))
    return pairs

def cell(s):
    return s.replace("|", "\\|").replace("\n", " ")

out = []
meta = script["meta"]
out.append(f"# {meta['title']['en']}\n\n# {meta['title']['zh']}\n")
out.append("""
## 网站结构（三层）

| 页面 | 路径 | 阅读时间 | 内容来源 |
|---|---|---|---|
| 影响力首页 | `index.html` | 30–60 秒（含 60 秒反例） | 文案直接写在 `index.html`（`.en` / `.zh` 成对元素）；交互在 `js/landing.js` |
| 四分钟导览 | `explainer/index.html` | 约 4 分钟，7 章 21 个 beat | `narration/explainer.json` |
| 技术交互导览 | `technical/index.html` | 约 10 分钟，10 章 43 个 beat | `narration/technical.json` |
| 核心图 / 分享图 | `figures/`、`assets/` | — | `tools/render.mjs` 生成 PNG / MP4 / GIF |

链接、引用、DOI 等集中在 `js/config.js`。
""")
out.append(f"网站内容文稿 · {meta['author']} · 解说音色：{meta['voices']['en']}（VoxCPM2 克隆）\n")
out.append("""
本文件由 `tools/export_content.py` 从网站的真实来源生成：解说词与“Read”段落来自 `narration/script.json`，屏幕文字来自 `js/scenes/ch*.js`，时长来自 `js/narration.js`。

**怎么改**

- 解说词 / 字幕 / Read 段落：直接改 `narration/script.json`（`en`、`zh`；若朗读文本需与字幕不同，用 `en_tts` / `zh_tts`；Read 段落是 HTML 片段，`<span class='f'>` 表示公式）。改完：删掉对应的 `.m4a`，运行 `python3 tools/make_audio.py && python3 tools/check_audio.py && python3 tools/build_narration.py`。
- 屏幕文字（画面里的标题、标签、注释）：在 `js/scenes/ch*.js` 里搜索对应的 `S.t("英文", "中文")`。
- 章节顺序与标题：`narration/script.json` 的 `chapters`（`id`、`title`、`kicker`）；动画场景按 `id` 对应 `js/scenes/`。
- 论文数字：不要手改，`python3 tools/extract_data.py` 从 `paper/numbers.tex` 与 `results/` 重新生成 `js/data.js`。

## 技术导览的结构总览（narration/technical.json）

| # | 章 Chapter | 引题 Kicker | beat 数 | 英文 / 中文时长 | 对应论文 | 画面 |
|---|---|---|---|---|---|---|""")
for i, ch in enumerate(script["chapters"]):
    de = sum(durs[(ch["id"], b["id"])].get("en", 0) for b in ch["beats"]); dz = sum(durs[(ch["id"], b["id"])].get("zh", 0) for b in ch["beats"])
    out.append(f"| {i} | {cell(ch['title']['zh'])} · {cell(ch['title']['en'])} | {cell(ch['kicker']['zh'])} · {cell(ch['kicker']['en'])} | {len(ch['beats'])} | {de/60:.1f} / {dz/60:.1f} 分 | {PAPER[ch['id']]} | {cell(VISUAL[ch['id']][0])} |")
tot_en = sum(d.get("en", 0) for d in durs.values()) / 60; tot_zh = sum(d.get("zh", 0) for d in durs.values()) / 60
out.append(f"\n合计：{sum(len(c['beats']) for c in script['chapters'])} 个 beat，英文 {tot_en:.1f} 分钟，中文 {tot_zh:.1f} 分钟。\n")
out.append("""
叙事线：识别（ch3–ch4）→ 校准（ch5–ch6）→ 设计（ch7）→ 真实数据（ch8）→ 原则（ch9），前面 ch0–ch2 建立问题与“协议价值”这个被评估的对象。每个 beat = 一句解说 + 一次画面变化；解说读完才推进到下一个 beat。
""")

for i, ch in enumerate(script["chapters"]):
    out.append(f"\n---\n\n## {i} · {ch['title']['zh']} / {ch['title']['en']}\n")
    out.append(f"- 引题：{ch['kicker']['zh']} / {ch['kicker']['en']}\n- 对应论文：{PAPER[ch['id']]}\n- 画面：{VISUAL[ch['id']][0]}\n- Visual: {VISUAL[ch['id']][1]}\n")
    out.append("### 解说词 Narration\n")
    for b in ch["beats"]:
        d = durs[(ch["id"], b["id"])]
        out.append(f"**{ch['id']}-{b['id']}**  (EN {d.get('en', 0):.0f} s · ZH {d.get('zh', 0):.0f} s)\n")
        out.append(f"- EN: {b['en']}")
        if b.get("en_tts"): out.append(f"  - EN 朗读文本: {b['en_tts']}")
        out.append(f"- ZH: {b['zh']}")
        if b.get("zh_tts"): out.append(f"  - ZH 朗读文本: {b['zh_tts']}")
        out.append("")
    pairs = screen_text(ch["id"])
    if pairs:
        out.append("### 屏幕文字 On-screen text\n\n| EN | ZH |\n|---|---|")
        for en, zh in pairs:
            out.append(f"| {cell(en)} | {cell(zh)} |")
        out.append("")
    out.append("### Read 段落（章下方的论文摘要）\n")
    out.append("**EN**\n\n" + html_to_md(ch["read"]["en"]) + "\n")
    out.append("**ZH**\n\n" + html_to_md(ch["read"]["zh"]) + "\n")

# ---- explainer narration
ex, exd = SETS["explainer"]
out.append("\n---\n\n# 四分钟导览的解说词（narration/explainer.json）\n")
for i, ch in enumerate(ex["chapters"]):
    out.append(f"\n## E{i} · {ch['title']['zh']} / {ch['title']['en']}\n\n- 引题：{ch['kicker']['zh']} / {ch['kicker']['en']}\n")
    for b in ch["beats"]:
        d = exd[(ch["id"], b["id"])]
        out.append(f"**{ch['id']}-{b['id']}**  (EN {d.get('en', 0):.0f} s · ZH {d.get('zh', 0):.0f} s)\n\n- EN: {b['en']}")
        if b.get("en_tts"): out.append(f"  - EN 朗读文本: {b['en_tts']}")
        out.append(f"- ZH: {b['zh']}")
        if b.get("zh_tts"): out.append(f"  - ZH 朗读文本: {b['zh_tts']}")
        out.append("")
    out.append("**Technical detail (EN)**\n\n" + html_to_md(ch["read"]["en"]) + "\n\n**技术细节（ZH）**\n\n" + html_to_md(ch["read"]["zh"]) + "\n")

out.append("""
---

## 页面固定文字（technical/index.html、explainer/index.html）

- 页眉眉题：Paper explainer · identification · calibration · observation design
- 标题：Counterfactual Evaluation of Temporal Observation Protocols
- 作者行：Xizhe Zhang
- 控件：English / 中文；Auto-advance chapters；Manuscript (PDF)
- 舞台提示：Press play to start the narration
- 键位说明：space play / pause · ← → beats · PgUp PgDn chapters · L language
- Read 段落标题：Read: this chapter in the paper / 读一读：这一章在论文里
- 页脚：Colour legend（latent trajectory Z · target Θ · realised protocol A · alternative protocol B · explained variance / value · non-identification）；Provenance（每个数字来自 paper/numbers.tex 与 results/）；Data（Sleep-EDF Expanded 与 Long-Term AF Database，仅用标注文件）
""")
(SITE / "CONTENT.md").write_text("\n".join(out))
print("wrote", SITE / "CONTENT.md", (SITE / "CONTENT.md").stat().st_size // 1024, "KB")
