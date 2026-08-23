#!/usr/bin/env node
/* Render the social card, the three key figures and a silent 30-second recording of the
 * counterexample, using headless Chrome over the DevTools protocol (no npm packages).
 *
 *   node tools/render.mjs              # everything, served from http://127.0.0.1:8791/
 *   node tools/render.mjs --only card  # card | figures | video
 *
 * Needs Google Chrome and ffmpeg.  Output: assets/social-card.png, figures/*.png,
 * assets/two-worlds.mp4 and assets/two-worlds.gif. */
import { spawn, execSync } from "node:child_process";
import { writeFileSync, mkdirSync, existsSync, rmSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const SITE = join(dirname(fileURLToPath(import.meta.url)), "..");
const BASE = process.env.CPV_BASE || "http://127.0.0.1:8791/";
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const onlyIdx = process.argv.indexOf("--only");
const only = (process.argv.find(a => a.startsWith("--only=")) || "").split("=")[1] || (onlyIdx >= 0 ? process.argv[onlyIdx + 1] : "");
const PORT = 9333;

const sleep = ms => new Promise(r => setTimeout(r, ms));
async function getJSON(url) { const r = await fetch(url); return r.json(); }

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map(); ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && this.pending.has(m.id)) { const { res, rej } = this.pending.get(m.id); this.pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result); } }; }
  send(method, params = {}) { return new Promise((res, rej) => { const id = ++this.id; this.pending.set(id, { res, rej }); this.ws.send(JSON.stringify({ id, method, params })); }); }
}
async function openTab(width, height, dpr) {
  const info = await getJSON(`http://127.0.0.1:${PORT}/json/new?about:blank`).catch(async () => { const r = await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" }); return r.json(); });
  const ws = new WebSocket(info.webSocketDebuggerUrl); await new Promise(r => ws.onopen = r);
  const c = new CDP(ws);
  await c.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: dpr, mobile: false });
  await c.send("Page.enable"); await c.send("Runtime.enable");
  return { c, ws, id: info.id };
}
async function navigate(c, url) {
  await c.send("Page.navigate", { url });
  for (let i = 0; i < 200; i++) { const r = await c.send("Runtime.evaluate", { expression: "!!window.CPV_READY && document.fonts.status === 'loaded'", returnByValue: true }); if (r.result.value) break; await sleep(100); }
  await sleep(400);
}
async function shot(c, path, clip) {
  const r = await c.send("Page.captureScreenshot", { format: "png", clip: clip ? { ...clip, scale: 1 } : undefined, captureBeyondViewport: true });
  writeFileSync(path, Buffer.from(r.data, "base64"));
}

const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, "--hide-scrollbars", "--no-first-run", "--disable-extensions", `--user-data-dir=${join(tmpdir(), "cpv-render-" + process.pid)}`, "about:blank"], { stdio: "ignore" });
try {
  for (let i = 0; i < 50; i++) { try { await getJSON(`http://127.0.0.1:${PORT}/json/version`); break; } catch (e) { await sleep(200); } }
  mkdirSync(join(SITE, "assets"), { recursive: true }); mkdirSync(join(SITE, "figures"), { recursive: true });

  if (!only || only === "card") {
    const { c } = await openTab(1200, 630, 2);
    await navigate(c, BASE + "assets/social.html");
    await shot(c, join(SITE, "assets/social-card.png"), { x: 0, y: 0, width: 1200, height: 630 });
    console.log("assets/social-card.png");
  }
  if (!only || only === "figures") {
    const names = { 1: "fig-two-worlds", 2: "fig-identify-calibrate-design", 3: "fig-resolution-evidence" };
    const { c } = await openTab(1600, 900, 2);
    for (const k of [1, 2, 3]) {
      await navigate(c, BASE + `figures/?render=${k}`);
      await shot(c, join(SITE, `figures/${names[k]}.png`), { x: 0, y: 0, width: 1600, height: 900 });
      console.log(`figures/${names[k]}.png`);
    }
  }
  if (!only || only === "video") {
    const fps = 15, seconds = 30, frames = fps * seconds, dir = join(tmpdir(), "cpv-frames-" + process.pid); mkdirSync(dir, { recursive: true });
    const { c } = await openTab(1600, 900, 1);
    await navigate(c, BASE + "figures/?render=1");
    for (let i = 0; i < frames; i++) {
      await c.send("Runtime.evaluate", { expression: `window.CPV_FRAME(${(i / fps).toFixed(4)})`, returnByValue: true });
      const r = await c.send("Page.captureScreenshot", { format: "png", clip: { x: 0, y: 0, width: 1600, height: 900, scale: 1 } });
      writeFileSync(join(dir, `f${String(i).padStart(4, "0")}.png`), Buffer.from(r.data, "base64"));
      if (i % 75 === 0) console.log(`frame ${i}/${frames}`);
    }
    execSync(`ffmpeg -loglevel error -y -framerate ${fps} -i "${dir}/f%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart "${join(SITE, "assets/two-worlds.mp4")}"`);
    execSync(`ffmpeg -loglevel error -y -i "${join(SITE, "assets/two-worlds.mp4")}" -vf "fps=10,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" "${join(SITE, "assets/two-worlds.gif")}"`);
    rmSync(dir, { recursive: true, force: true });
    console.log("assets/two-worlds.mp4, assets/two-worlds.gif");
  }
} finally {
  chrome.kill("SIGKILL");
}
