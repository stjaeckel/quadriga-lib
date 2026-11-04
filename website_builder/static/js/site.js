/* Optional extras for the Quadriga-Lib site. Every page works without this file. */
(function () {
  "use strict";
  var store = {
    get: function (k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

  /* 1. Copy buttons on code blocks */
  document.querySelectorAll("pre").forEach(function (pre) {
    if (pre.closest(".out")) return;
    var box = pre.parentElement.classList.contains("code") ? pre.parentElement : null;
    if (!box) { box = document.createElement("div"); box.className = "code"; pre.parentNode.insertBefore(box, pre); box.appendChild(pre); }
    var b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "Copy";
    b.addEventListener("click", function () {
      var t = pre.innerText.replace(/\n$/, "");
      (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(function () {
        b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1500);
      }, function () { b.textContent = "Select and copy"; });
    });
    box.appendChild(b);
  });

  /* 2. Remember the chosen language in code tabs (stored only after a click) */
  var saved = store.get("qlib-lang");
  document.querySelectorAll("[data-lang-tabs]").forEach(function (tabs) {
    if (saved) { var r = tabs.querySelector('input[value="' + saved + '"]'); if (r) r.checked = true; }
    tabs.addEventListener("change", function (e) { if (e.target.value) store.set("qlib-lang", e.target.value); });
  });

  /* 3. Function filter in the API index, "/" focuses it */
  document.querySelectorAll(".idx-filter").forEach(function (filter) {
    var nav = filter.closest(".idx");
    var groups = Array.prototype.slice.call(nav.querySelectorAll("details"));
    filter.addEventListener("input", function () {
      var q = filter.value.trim().toLowerCase();
      groups.forEach(function (d) {
        var hit = 0;
        d.querySelectorAll("li.fn").forEach(function (li) {
          var ok = !q || li.textContent.toLowerCase().indexOf(q) >= 0;
          li.hidden = !ok; if (ok) hit++;
        });
        d.parentElement.hidden = q && !hit;
        if (q) d.open = hit > 0;
      });
    });
    filter.addEventListener("keydown", function (e) {
      if (e.key === "Enter") { var a = nav.querySelector("li.fn:not([hidden]) a"); if (a) { location.hash = a.getAttribute("href"); } }
    });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key !== "/" || /INPUT|TEXTAREA/.test(document.activeElement.tagName)) return;
    var wide = window.matchMedia("(min-width: 960px)").matches;
    var f = document.querySelector(wide ? ".doc-side .idx-filter" : ".doc-index .idx-filter");
    if (!f) return;
    e.preventDefault(); var d = f.closest("details.doc-index"); if (d) d.open = true; f.focus();
  });

  /* 4. Highlight the function being read and open its section in the sidebar */
  var links = {};
  document.querySelectorAll(".doc-side a[href^='#']").forEach(function (a) { links[decodeURIComponent(a.getAttribute("href").slice(1))] = a; });
  var entries = document.querySelectorAll(".fn-entry, .api-section, .prose h2[id], .prose h3[id]");
  if ("IntersectionObserver" in window && entries.length) {
    var current = null;
    var io = new IntersectionObserver(function (list) {
      list.forEach(function (en) {
        if (!en.isIntersecting) return;
        var a = links[en.target.id]; if (!a || a === current) return;
        if (current) current.classList.remove("current");
        a.classList.add("current"); current = a;
        var d = a.closest("details"); if (d) d.open = true;
      });
    }, { rootMargin: "-110px 0px -70% 0px" });
    entries.forEach(function (el) { io.observe(el); });
  }
})();
