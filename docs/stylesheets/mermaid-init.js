// Initialise Mermaid with a theme that matches the page colour scheme.
// Runs after the page loads so it can detect the active palette.
document.addEventListener("DOMContentLoaded", function () {
  var isDark =
    document.body.getAttribute("data-md-color-scheme") === "slate";

  mermaid.initialize({
    startOnLoad: true,
    theme: isDark ? "dark" : "default",
    themeVariables: isDark
      ? {}
      : {
          primaryColor: "#6750a4",
          primaryTextColor: "#ffffff",
          primaryBorderColor: "#4a3880",
          lineColor: "#555",
          secondaryColor: "#ede7f6",
          tertiaryColor: "#f3e5f5",
          background: "#ffffff",
          mainBkg: "#ffffff",
          nodeBorder: "#6750a4",
          clusterBkg: "#f5f0ff",
          titleColor: "#1a1a2e",
          edgeLabelBackground: "#ffffff",
          fontFamily: "Inter, sans-serif",
        },
  });

  // Re-render if the user toggles light/dark mode
  var observer = new MutationObserver(function () {
    var dark =
      document.body.getAttribute("data-md-color-scheme") === "slate";
    mermaid.initialize({
      startOnLoad: false,
      theme: dark ? "dark" : "default",
    });
    // Force re-render of all diagrams
    document.querySelectorAll(".mermaid").forEach(function (el) {
      el.removeAttribute("data-processed");
    });
    mermaid.run();
  });

  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
});
