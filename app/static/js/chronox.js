/* ChronoX Interactive Frontend Controller */

(function () {
  "use strict";

  const htmlEl = document.documentElement;
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');
  const csrfToken = csrfMeta ? csrfMeta.getAttribute("content") : "";

  /* --- Theme Persistence --- */
  function applyTheme(theme) {
    htmlEl.setAttribute("data-theme", theme);
    localStorage.setItem("chronox-theme", theme);
    document.querySelectorAll("[data-theme-icon]").forEach((icon) => {
      icon.className = theme === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
    });
    window.dispatchEvent(new CustomEvent("chronox:themechange", { detail: { theme } }));
  }

  const savedTheme = localStorage.getItem("chronox-theme") || htmlEl.getAttribute("data-theme") || "light";
  applyTheme(savedTheme);

  document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const next = htmlEl.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      if (btn.dataset.authenticated === "true" && csrfToken) {
        fetch("/profile/theme", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": csrfToken
          },
          body: JSON.stringify({ theme: next })
        }).catch(() => {});
      }
      showToast(`Switched to ${next === "dark" ? "Dark" : "Light"} mode`, "info");
    });
  });

  /* --- Toast Notification Helper --- */
  window.showToast = function (message, tone) {
    const container = document.getElementById("cxToastContainer");
    if (!container) return;
    const toneClass = tone === "danger" ? "text-bg-danger" : tone === "success" ? "text-bg-success" : "text-bg-dark";
    const toastEl = document.createElement("div");
    toastEl.className = `toast align-items-center ${toneClass} border-0`;
    toastEl.setAttribute("role", "status");
    toastEl.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">${message}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>`;
    container.appendChild(toastEl);
    if (window.bootstrap && window.bootstrap.Toast) {
      const toast = new window.bootstrap.Toast(toastEl, { delay: 3200 });
      toast.show();
      toastEl.addEventListener("hidden.bs.toast", () => toastEl.remove());
    }
  };

  /* --- Animated Number Counters --- */
  function animateCounters() {
    const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.querySelectorAll("[data-count-to]").forEach((el) => {
      const target = parseFloat(el.dataset.countTo);
      const decimals = parseInt(el.dataset.decimals || "0", 10);
      const suffix = el.dataset.suffix || "";
      if (isNaN(target)) return;
      if (prefersReduced) {
        el.textContent = target.toFixed(decimals) + suffix;
        return;
      }
      const duration = 900;
      const start = performance.now();
      function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = (target * eased).toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }
  animateCounters();

  /* --- Password Visibility Toggles --- */
  document.querySelectorAll("[data-toggle-password]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-toggle-password");
      const input = document.getElementById(targetId);
      if (!input) return;
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btn.innerHTML = isPassword ? '<i class="bi bi-eye-slash"></i>' : '<i class="bi bi-eye"></i>';
      btn.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
    });
  });

  /* --- Validation Shake on Invalid Submit --- */
  document.querySelectorAll("form.needs-shake").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
        form.classList.remove("cx-shake");
        void form.offsetWidth;
        form.classList.add("cx-shake");
        form.classList.add("was-validated");
        showToast("Please review highlighted fields before continuing.", "danger");
      }
    });
  });

  /* --- Initialize Bootstrap Tooltips --- */
  if (window.bootstrap && window.bootstrap.Tooltip) {
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => {
      new window.bootstrap.Tooltip(el);
    });
  }

  /* --- Initialize AOS Scroll Animations --- */
  if (window.AOS) {
    window.AOS.init({
      duration: 600,
      once: true,
      disable: () => window.matchMedia("(prefers-reduced-motion: reduce)").matches
    });
  }

  /* --- Register Service Worker for PWA --- */
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    });
  }
})();
