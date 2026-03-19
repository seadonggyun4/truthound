document.addEventListener("DOMContentLoaded", () => {
  const root = document.documentElement;
  const body = document.body;
  const defaultPalette = {
    color: {
      media: "",
      scheme: "slate",
      primary: "indigo",
      accent: "indigo",
    },
  };

  if (typeof window.__md_get === "function" && typeof window.__md_set === "function") {
    const palette = window.__md_get("__palette");
    if (!palette || !palette.color) {
      window.__md_set("__palette", defaultPalette);
    }
  }

  // Avoid animating the initial paint; enable transitions after the page settles.
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      root.classList.add("theme-ready");
    });
  });

  if (!body) {
    return;
  }

  let switchTimer;
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.attributeName !== "data-md-color-scheme") {
        continue;
      }
      root.classList.add("theme-switching");
      clearTimeout(switchTimer);
      switchTimer = window.setTimeout(() => {
        root.classList.remove("theme-switching");
      }, 380);
    }
  });

  observer.observe(body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
});
