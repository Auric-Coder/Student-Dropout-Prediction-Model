/* =============================================
   Form submit — loading state
   ============================================= */
const form = document.getElementById("predictForm");
const btn  = document.getElementById("submitBtn");

if (form && btn) {
  form.addEventListener("submit", () => {
    btn.classList.add("loading");
    const txt = btn.querySelector(".btn-text");
    if (txt) txt.textContent = "Analyzing…";
  });
}

/* =============================================
   Live value indicator + colour feedback
   for numeric indicator inputs
   ============================================= */
document.querySelectorAll("input[type='number']").forEach((el) => {
  const viId = "vi-" + el.id;
  const vi   = document.getElementById(viId);
  const min  = parseFloat(el.dataset.min ?? el.min ?? 0);
  const max  = parseFloat(el.dataset.max ?? el.max ?? 100);

  function update() {
    const v = parseFloat(el.value);
    if (isNaN(v) || el.value === "") {
      if (vi) vi.textContent = "";
      el.style.borderColor = "";
      return;
    }
    const pct = Math.max(0, Math.min(1, (v - min) / (max - min)));
    // colour scale: low=amber, mid=green, high=red (for risk-style fields)
    if (pct < 0.4) {
      el.style.borderColor = "#f59e0b";
      if (vi) { vi.textContent = v; vi.style.color = "#f59e0b"; }
    } else if (pct < 0.75) {
      el.style.borderColor = "#10b981";
      if (vi) { vi.textContent = v; vi.style.color = "#10b981"; }
    } else {
      el.style.borderColor = "#10b981";
      if (vi) { vi.textContent = v; vi.style.color = "#10b981"; }
    }
  }

  el.addEventListener("input", update);
  el.addEventListener("focus", () => el.select && el.select());
  update();
});

/* =============================================
   Result page animations
   ============================================= */
document.addEventListener("DOMContentLoaded", () => {

  /* --- Animated probability bars --- */
  document.querySelectorAll(".prob-fill").forEach((el) => {
    const target = el.style.width || "0%";
    el.style.width = "0%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.style.width = target; });
    });
  });

  /* --- Animated indicator mini-bars --- */
  document.querySelectorAll(".ind-bar").forEach((el) => {
    const target = el.style.width || "0%";
    el.style.width = "0%";
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.style.width = target; });
    });
  });

  /* --- SVG gauge arc animation --- */
  const arc = document.querySelector(".gauge-arc");
  if (arc) {
    const finalOffset = parseFloat(arc.getAttribute("data-target-offset"));
    const dasharray   = parseFloat(arc.getAttribute("stroke-dasharray"));
    arc.style.strokeDashoffset = dasharray;  // start at 0% fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        arc.style.transition = "stroke-dashoffset 1.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)";
        arc.style.strokeDashoffset = finalOffset;
      });
    });
  }

  /* --- Counter animation for the gauge number --- */
  const gaugeNum = document.getElementById("gaugeNum");
  if (gaugeNum) {
    const target  = parseFloat(gaugeNum.dataset.target ?? 0);
    const dur     = 1300;   // ms, matches arc transition
    const start   = performance.now();

    function tick(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / dur, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      gaugeNum.textContent = (eased * target).toFixed(1);
      if (progress < 1) requestAnimationFrame(tick);
      else gaugeNum.textContent = target.toFixed(1);
    }

    requestAnimationFrame(tick);
  }
});
