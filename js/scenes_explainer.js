/* Four-minute story: seven chapters built from the technical scenes.
 *
 * Each explainer chapter wraps one or more technical scenes and re-groups their
 * beats under the shorter narration.  A wrapper keeps the wrapped scene's
 * state and drawing; a composite switches between two wrapped scenes
 * mid-chapter (and re-applies their WebGL use). */
(function () {
  const CPV = window.CPV, T = CPV.scenes;

  /* Activate sub-scene `sub` on `stage` as part of composite `host`. */
  function activate(host, stage, sub) {
    if (host._active === sub) return;
    if (host._active && host._active.leave) host._active.leave(stage);
    stage.clearAnimations(); stage.field.clearLayers();
    if (!sub._ready) { sub.setup(stage); sub._ready = true; }
    if (sub.enter) sub.enter(stage);
    host._active = sub; host.useGL = !!sub.useGL; host.glRect = sub.glRect || null;
    stage.applySceneGL();
  }
  function wrap(id, sub, beats) {
    return {
      id, _active: null,
      setup() {},
      enter(stage) { this._active = null; activate(this, stage, sub); },
      leave(stage) { if (this._active && this._active.leave) this._active.leave(stage); this._active = null; },
      beats: beats.map(fn => function (stage, player) { fn.call(this, stage, player); }),
      draw(stage, t, dt) { if (this._active) this._active.draw(stage, t, dt); },
      onPointer(stage, type, x, y, e) { return this._active && this._active.onPointer ? this._active.onPointer(stage, type, x, y, e) : false; },
    };
  }
  /* Run technical beat j of scene `sub` on this host (activating it first). */
  const run = (sub, j) => function (stage) { activate(this, stage, sub); sub.beats[j].call(sub, stage, window.player); };
  /* Run beat j, then beat k after a delay. */
  const chain = (sub, j, k, ms) => function (stage) { activate(this, stage, sub); sub.beats[j].call(sub, stage, window.player); stage.delay(ms, () => sub.beats[k].call(sub, stage, window.player)); };

  // e0 — the question: the trajectory field, then A's ticks and B's never-run ticks
  CPV.scenes.e0 = wrap("e0", T.ch0, [
    function (stage) { activate(this, stage, T.ch0); T.ch0.beats[0].call(T.ch0, stage); stage.delay(2600, () => T.ch0.beats[1].call(T.ch0, stage)); },
    run(T.ch0, 2),
  ]);
  // e1 — two worlds (Theorem 3)
  CPV.scenes.e1 = wrap("e1", T.ch3, [
    run(T.ch3, 0),
    chain(T.ch3, 1, 2, 3600),
    run(T.ch3, 3),
    function (stage) { activate(this, stage, T.ch3); T.ch3.beats[4].call(T.ch3, stage); stage.delay(2600, () => T.ch3.beats[5].call(T.ch3, stage)); },
  ]);
  // e2 — value-specific identification: the definition (ch2), then the three blocks and Theorem 5 (ch4)
  CPV.scenes.e2 = wrap("e2", T.ch2, [
    run(T.ch2, 0),
    chain(T.ch4, 0, 1, 4200),
    run(T.ch4, 2),
  ]);
  // e3 — what additional data resolve (ch5): augmentation, dense calibration, rates
  CPV.scenes.e3 = wrap("e3", T.ch5, [
    run(T.ch5, 0),
    chain(T.ch5, 1, 2, 5200),
    function (stage) { activate(this, stage, T.ch5); T.ch5.beats[3].call(T.ch5, stage); stage.delay(5200, () => T.ch5.beats[4].call(T.ch5, stage)); },
  ]);
  // e4 — resolution (ch6)
  CPV.scenes.e4 = wrap("e4", T.ch6, [
    run(T.ch6, 0),
    chain(T.ch6, 1, 2, 5200),
    run(T.ch6, 3),
  ]);
  // e5 — design (ch7) and evidence (ch8)
  CPV.scenes.e5 = wrap("e5", T.ch7, [
    function (stage) { activate(this, stage, T.ch7); T.ch7.beats[0].call(T.ch7, stage); stage.delay(2400, () => T.ch7.beats[1].call(T.ch7, stage)); stage.delay(5600, () => T.ch7.beats[2].call(T.ch7, stage)); },
    run(T.ch7, 3),
    chain(T.ch8, 1, 2, 7000),
    run(T.ch8, 3),
  ]);
  // e6 — takeaway (ch9)
  CPV.scenes.e6 = wrap("e6", T.ch9, [
    run(T.ch9, 0),
    chain(T.ch9, 1, 2, 5200),
  ]);
})();
