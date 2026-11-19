/* Shared by the four decks; build_standalone.py inlines it. Loaded after each deck's own script, which
   defines goTo(index) and writes #<number> into the URL.
   - Stable links: every slide has data-slug, so deck.html#clip-higher keeps working when slides move.
   - Theme: T (or the half-moon button) switches between the dark design and a light version.
   - Print: every fragment is revealed before printing, so a PDF handout shows complete slides. */
(function () {
  const slides = Array.from(document.querySelectorAll("section.slide"));
  const bySlug = {};
  slides.forEach((s, i) => { if (s.dataset.slug) bySlug[s.dataset.slug] = i; });

  const slugOf = (hash) => decodeURIComponent((hash || "").slice(1));
  function jumpTo(slug) {
    if (slug in bySlug && typeof goTo === "function") goTo(bySlug[slug]);
  }
  window.addEventListener("hashchange", () => jumpTo(slugOf(location.hash)));
  // Each deck builds its navigation in a DOMContentLoaded handler registered earlier, and that handler
  // rewrites the URL to #1; so remember the slug now and jump once the deck is ready.
  const initialSlug = slugOf(location.hash);
  const jumpInitial = () => setTimeout(() => jumpTo(initialSlug), 0);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", jumpInitial);
  else jumpInitial();

  const root = document.documentElement;
  const store = {
    get() { try { return localStorage.getItem("deckTheme"); } catch (e) { return null; } },
    set(v) { try { localStorage.setItem("deckTheme", v); } catch (e) { /* file:// in some browsers */ } },
  };
  if (store.get() === "light") root.dataset.theme = "light";
  function toggleTheme() {
    root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
    store.set(root.dataset.theme);
  }
  const controls = document.querySelector(".controls");
  if (controls) {
    const btn = document.createElement("button");
    btn.id = "themeBtn";
    btn.type = "button";
    btn.textContent = "\u25D0";
    btn.title = "Light / dark theme (T)";
    btn.setAttribute("aria-label", "Toggle light or dark theme");
    btn.addEventListener("click", toggleTheme);
    controls.appendChild(btn);
  }
  document.addEventListener("keydown", (e) => {
    const typing = e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (!typing && (e.key === "t" || e.key === "T")) toggleTheme();
  });

  window.addEventListener("beforeprint", () => {
    document.querySelectorAll(".fragment").forEach((f) => f.classList.add("visible"));
  });
})();
