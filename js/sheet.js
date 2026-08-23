/* Storyboard mode: index.html?sheet=all  (or ?sheet=ch3,ch4)
 * Renders the final frame of every beat into a grid with its narration, for
 * review and for printing a storyboard.  Animations are fast-forwarded, no
 * audio is played.  Add &lang=zh for the Chinese script. */
(function () {
  const q = new URLSearchParams(location.search), want = q.get("sheet");
  if (!want) return;
  window.CPV_SHEET = true;
  const CPV = window.CPV;
  CPV.motion = 0.0005;
  document.body.classList.add("sheet-mode");
  const css = document.createElement("style");
  css.textContent = `
    body.sheet-mode .top, body.sheet-mode .site-foot, body.sheet-mode .chapters, body.sheet-mode .transport, body.sheet-mode .transcript, body.sheet-mode .read, body.sheet-mode .chapter-head { display: none !important; }
    body.sheet-mode .theatre { display: block; min-height: 0; }
    body.sheet-mode .stage-col { padding: 0; }
    body.sheet-mode .stage { position: fixed; left: -4000px; top: 0; width: 1600px; height: 900px; aspect-ratio: auto; max-height: none; }
    #sheet { padding: 24px; display: grid; grid-template-columns: repeat(var(--cols, 2), 1fr); gap: 22px 22px; }
    #sheet h2 { grid-column: 1 / -1; font-family: var(--font-display); font-weight: 500; font-size: 22px; margin: 14px 0 0; color: var(--ink); }
    #sheet figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    #sheet figure img { display: block; width: 100%; aspect-ratio: 16 / 9; background: #0B1119; }
    #sheet figcaption { padding: 10px 14px 12px; font-size: 13px; line-height: 1.45; color: var(--ink-2); }
    #sheet figcaption b { font-family: var(--font-mono); color: var(--muted); font-weight: 500; margin-right: 8px; }
  `;
  document.head.appendChild(css);
  const grid = document.createElement("div"); grid.id = "sheet";
  grid.style.setProperty("--cols", q.get("cols") || "2");
  document.body.appendChild(grid);

  /* Drive the stage synchronously: timers are short-circuited in sheet mode and tweens finish in one frame,
   * so a few manual frames reach each beat's final state without waiting on requestAnimationFrame
   * (which does not fire at all in a hidden tab). */
  let clock = performance.now();
  const frames = n => { const stage = window.player.stage; for (let k = 0; k < n; k++) { clock += 16; stage.frame(clock); } return Promise.resolve(); };

  window.addEventListener("load", async () => {
    const player = window.player, stage = player.stage;
    if (q.get("lang")) player.setLang(q.get("lang"));
    player.autoplay = false; player.playing = false;
    const chapters = player.chapters, wanted = want === "all" ? chapters.map(c => c.id) : want.split(",");
    for (let i = 0; i < chapters.length; i++) {
      const ch = chapters[i]; if (!wanted.includes(ch.id)) continue;
      const h = document.createElement("h2"); h.textContent = `${ch.id} · ${player.t(ch.title)}`; grid.appendChild(h);
      player.gotoChapter(i, { play: false });
      for (let j = 0; j < ch.beats.length; j++) {
        player.playing = false; player.showBeat(j, { silent: true });
        await frames(14);
        const c = document.createElement("canvas"); c.width = stage.ovCanvas.width; c.height = stage.ovCanvas.height;
        const cx = c.getContext("2d"); cx.fillStyle = CPV.C.bg; cx.fillRect(0, 0, c.width, c.height);
        if (stage.scene.useGL && stage.field.ok) cx.drawImage(stage.glCanvas, 0, 0);
        cx.drawImage(stage.ovCanvas, 0, 0);
        const fig = document.createElement("figure");
        const img = new Image(); img.src = c.toDataURL("image/png"); fig.appendChild(img);
        const cap = document.createElement("figcaption"); cap.innerHTML = `<b>${ch.id}-${ch.beats[j].id}</b>${player.t(ch.beats[j])}`; fig.appendChild(cap);
        grid.appendChild(fig);
      }
    }
    document.title = "Storyboard · " + document.title;
    window.CPV_SHEET_DONE = true;
  });
})();
