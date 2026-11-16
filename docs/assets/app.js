/* AI Engineer Roadmap — docs site runtime
   Theme switching, search, code copy, mobile nav, reading progress. */
(function () {
  "use strict";

  var root = document.documentElement.dataset.root || "";
  var THEMES = [
    ["github-light", "GitHub light"],
    ["github-dark", "GitHub dark"],
    ["book", "Book (paper)"],
    ["slate", "Slate (dark)"],
    ["terminal", "Terminal"]
  ];

  /* ---------- theme ---------- */
  function applyTheme(name) {
    document.documentElement.setAttribute("data-theme", name);
    try { localStorage.setItem("aer-theme", name); } catch (e) {}
    var sel = document.getElementById("theme");
    if (sel) sel.value = name;
  }

  function initTheme() {
    var saved = null;
    try { saved = localStorage.getItem("aer-theme"); } catch (e) {}
    var sel = document.getElementById("theme");
    if (sel) {
      sel.innerHTML = "";
      THEMES.forEach(function (t) {
        var o = document.createElement("option");
        o.value = t[0];
        o.textContent = t[1];
        sel.appendChild(o);
      });
      sel.addEventListener("change", function () { applyTheme(sel.value); });
    }
    if (saved) {
      applyTheme(saved);
    } else if (sel) {
      sel.value = document.documentElement.getAttribute("data-theme") || "github-light";
    }
  }

  /* ---------- sidebar ---------- */
  function initNav() {
    var btn = document.getElementById("menuBtn");
    if (btn) {
      btn.addEventListener("click", function () {
        document.body.classList.toggle("nav-open");
      });
    }
    document.querySelectorAll(".nav-stage").forEach(function (b) {
      b.addEventListener("click", function () {
        var g = b.closest(".nav-group");
        if (g) g.classList.toggle("open");
      });
    });
    var active = document.querySelector(".nav-list a.active");
    if (active) {
      var g = active.closest(".nav-group");
      if (g) g.classList.add("open");
      var sb = document.getElementById("sidebar");
      if (sb && active.offsetTop > sb.clientHeight * 0.8) {
        sb.scrollTop = Math.max(0, active.offsetTop - sb.clientHeight / 2);
      }
    }
    document.addEventListener("click", function (e) {
      if (document.body.classList.contains("nav-open")) {
        var sb = document.getElementById("sidebar");
        if (sb && !sb.contains(e.target) && e.target.id !== "menuBtn") {
          document.body.classList.remove("nav-open");
        }
      }
    });
  }

  /* ---------- code blocks ---------- */
  function initCode() {
    document.querySelectorAll(".code-block pre").forEach(function (pre) {
      var block = pre.parentElement;
      var code = pre.querySelector("code");
      var lang = "";
      if (code) {
        var m = (code.className || "").match(/language-([\w+#-]+)/);
        if (m) lang = m[1];
      }
      if (lang) {
        var tag = document.createElement("span");
        tag.className = "lang";
        tag.textContent = lang;
        block.appendChild(tag);
      }
      var btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.textContent = "Copy";
      btn.addEventListener("click", function () {
        var text = code ? code.innerText : pre.innerText;
        var done = function () {
          btn.textContent = "Copied";
          btn.classList.add("done");
          setTimeout(function () {
            btn.textContent = "Copy";
            btn.classList.remove("done");
          }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {});
        } else {
          var ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand("copy"); done(); } catch (e) {}
          document.body.removeChild(ta);
        }
      });
      block.appendChild(btn);
    });
  }

  /* ---------- reading progress + to top ---------- */
  function initProgress() {
    var bar = document.getElementById("progress");
    var top = document.getElementById("toTop");
    if (top) {
      top.addEventListener("click", function () {
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    }
    var tick = function () {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      var pct = max > 0 ? (h.scrollTop / max) * 100 : 0;
      if (bar) bar.style.width = pct + "%";
      if (top) top.classList.toggle("show", h.scrollTop > 900);
    };
    window.addEventListener("scroll", tick, { passive: true });
    window.addEventListener("resize", tick);
    tick();
  }

  /* ---------- search ---------- */
  function initSearch() {
    var input = document.getElementById("q");
    var box = document.getElementById("results");
    if (!input || !box) return;
    var index = null;
    var loading = false;
    var sel = -1;

    function load() {
      if (index || loading) return;
      loading = true;
      fetch(root + "assets/search-index.json")
        .then(function (r) { return r.json(); })
        .then(function (data) { index = data; })
        .catch(function () { index = []; });
    }

    function score(item, terms) {
      var t = item.t.toLowerCase();
      var x = (item.x || "").toLowerCase();
      var s = 0;
      terms.forEach(function (term) {
        if (t.indexOf(term) === 0) s += 40;
        else if (t.indexOf(term) > -1) s += 25;
        if (x.indexOf(term) > -1) s += 3;
      });
      return s;
    }

    function render(items, terms) {
      box.innerHTML = "";
      sel = -1;
      if (!items.length) {
        box.innerHTML = '<div class="empty">No lessons match that.</div>';
        box.classList.add("open");
        return;
      }
      items.slice(0, 12).forEach(function (it) {
        var a = document.createElement("a");
        a.href = root + it.u;
        a.innerHTML =
          '<span class="r-title">' + it.t + '</span>' +
          '<span class="r-stage">' + it.s + '</span>';
        box.appendChild(a);
      });
      box.classList.add("open");
    }

    input.addEventListener("input", function () {
      var q = input.value.trim().toLowerCase();
      if (q.length < 2) { box.classList.remove("open"); return; }
      load();
      if (!index) { box.innerHTML = '<div class="empty">Loading…</div>'; box.classList.add("open"); return; }
      var terms = q.split(/\s+/);
      var hits = index
        .map(function (it) { return { it: it, s: score(it, terms) }; })
        .filter(function (h) { return h.s > 0; })
        .sort(function (a, b) { return b.s - a.s; })
        .map(function (h) { return h.it; });
      render(hits, terms);
    });

    input.addEventListener("keydown", function (e) {
      var links = box.querySelectorAll("a");
      if (e.key === "ArrowDown") {
        e.preventDefault();
        sel = Math.min(sel + 1, links.length - 1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        sel = Math.max(sel - 1, 0);
      } else if (e.key === "Enter") {
        if (sel > -1 && links[sel]) { window.location.href = links[sel].href; }
        return;
      } else if (e.key === "Escape") {
        box.classList.remove("open");
        input.blur();
        return;
      } else {
        return;
      }
      links.forEach(function (l, i) { l.classList.toggle("sel", i === sel); });
      if (links[sel]) links[sel].scrollIntoView({ block: "nearest" });
    });

    document.addEventListener("click", function (e) {
      if (!box.contains(e.target) && e.target !== input) box.classList.remove("open");
    });

    document.addEventListener("keydown", function (e) {
      if ((e.key === "/" || (e.key === "k" && (e.metaKey || e.ctrlKey))) &&
          document.activeElement !== input) {
        e.preventDefault();
        input.focus();
        input.select();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    initTheme();
    initNav();
    initCode();
    initProgress();
    initSearch();
  });
})();
