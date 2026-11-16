/* Add independent disclosure controls without changing API links or anchors. */
(() => {
  function enhanceNavigation() {
    if (!document.querySelector(".doc-object")) return;

    const sidebar = document.querySelector(".md-nav--primary");
    if (!sidebar) return;

    sidebar.querySelectorAll("li.md-nav__item").forEach((item, index) => {
      const children = item.querySelector(":scope > nav");
      const link = item.querySelector(":scope > a.md-nav__link");
      if (!children || !link || item.querySelector(":scope > .api-nav-row")) return;
      if (!children.matches(".md-nav--secondary") && !item.closest(".md-nav--secondary")) return;

      const row = document.createElement("div");
      row.className = "api-nav-row";
      link.before(row);
      row.append(link);

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "api-nav-toggle";
      children.id ||= `api-nav-children-${index}`;
      toggle.setAttribute("aria-controls", children.id);
      row.append(toggle);

      const label = link.textContent.trim();
      function setExpanded(expanded) {
        toggle.setAttribute("aria-expanded", String(expanded));
        toggle.setAttribute("aria-label", `${expanded ? "Collapse" : "Expand"} ${label}`);
        children.classList.toggle("api-nav-collapsed", !expanded);
      }
      setExpanded(children.matches(".md-nav--secondary"));
      toggle.addEventListener("click", () => {
        setExpanded(toggle.getAttribute("aria-expanded") !== "true");
      });
    });
    revealAnchor();
  }

  function revealAnchor() {
    if (!location.hash) return;
    document.querySelectorAll(".md-nav--primary .api-nav-row").forEach((row) => {
      const toggle = row.querySelector("button");
      const children = document.getElementById(toggle.getAttribute("aria-controls"));
      const links = [row.querySelector("a"), ...children.querySelectorAll("a")];
      if (links.some((link) => link.hash === location.hash)) {
        children.classList.remove("api-nav-collapsed");
        toggle.setAttribute("aria-expanded", "true");
        toggle.setAttribute("aria-label", `Collapse ${row.querySelector("a").textContent.trim()}`);
      }
    });
  }

  // The theme emits document$ after client-side navigation as well as initial load.
  if (typeof document$ !== "undefined") document$.subscribe(enhanceNavigation);
  else enhanceNavigation();
  window.addEventListener("hashchange", revealAnchor);
})();
