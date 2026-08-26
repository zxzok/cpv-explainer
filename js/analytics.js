/* Privacy-friendly visit statistics via GoatCounter (no cookies, GDPR-friendly).
 * Enabled only when CPV_LINKS.goatcounter is set to the site code (https://<code>.goatcounter.com).
 * Counts page views automatically and forwards the landing page's data-track events as event hits. */
(function () {
  var L = window.CPV_LINKS || {};
  if (!L.goatcounter) return;
  var s = document.createElement("script");
  s.async = true; s.src = "https://gc.zgo.at/count.js";
  s.setAttribute("data-goatcounter", "https://" + L.goatcounter + ".goatcounter.com/count");
  document.head.appendChild(s);
  document.addEventListener("click", function (ev) {
    var el = ev.target && ev.target.closest && ev.target.closest("[data-track]");
    if (!el || !window.goatcounter || !window.goatcounter.count) return;
    window.goatcounter.count({ path: "event/" + el.getAttribute("data-track"), title: el.getAttribute("data-track"), event: true });
  }, { capture: true, passive: true });
})();
