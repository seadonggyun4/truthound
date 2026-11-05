/* Collect public documentation traffic only on the production host. */
(function () {
  if (window.location.hostname !== "www.truthound-docs.tech") return;
  if (window.__truthoundInsightsLoaded) return;
  window.__truthoundInsightsLoaded = true;

  window.va = window.va || function () {
    (window.vaq = window.vaq || []).push(arguments);
  };
  window.si = window.si || function () {
    (window.siq = window.siq || []).push(arguments);
  };
  // Exclude query strings and fragments, including documentation search terms.
  function publicPage(event) {
    var url = new URL(event.url, window.location.origin);
    url.search = "";
    url.hash = "";
    return Object.assign({}, event, { url: url.href });
  }
  window.va("beforeSend", publicPage);
  window.si("beforeSend", publicPage);
  ["/_vercel/insights/script.js", "/_vercel/speed-insights/script.js"].forEach(function (src) {
    var script = document.createElement("script");
    script.defer = true;
    script.src = src;
    document.head.appendChild(script);
  });
})();
