(function () {
  var path = window.location.pathname;
  if (/(^|\/)(ko|en)(\/|$)/.test(path)) {
    return;
  }

  var storageKey = "truthound_docs_lang";
  var stored = window.localStorage.getItem(storageKey);
  var browserLanguages = navigator.languages || [navigator.language || ""];
  var browserLang = browserLanguages.some(function (lang) {
    return String(lang).toLowerCase().indexOf("ko") === 0;
  })
    ? "ko"
    : "en";
  var lang = stored === "ko" || stored === "en" ? stored : browserLang;
  var nextPath = path.replace(/\/$/, "") + "/" + lang + "/";

  window.localStorage.setItem(storageKey, lang);
  window.location.replace(nextPath.replace(/\/{2,}/g, "/"));
})();
