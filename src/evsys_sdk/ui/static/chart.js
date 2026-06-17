// Minimal dependency-free multi-series line chart on a <canvas>.
// drawLineChart(canvas, series) where series = [{label, color, points:[[x,y]...]}].
(function () {
  const PAD = { l: 46, r: 10, t: 8, b: 22 };

  function bounds(series) {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of series) for (const [x, y] of s.points) {
      if (x == null || y == null || !isFinite(x) || !isFinite(y)) continue;
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
    if (!isFinite(xmin)) { xmin = 0; xmax = 1; ymin = 0; ymax = 1; }
    if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
    if (ymin === ymax) { ymin -= 0.5; ymax += 0.5; }
    const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad;
    return { xmin, xmax, ymin, ymax };
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    return (+v.toFixed(4)).toString();
  }

  function drawLineChart(canvas, series) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 320, cssH = canvas.clientHeight || 200;
    canvas.width = cssW * dpr; canvas.height = cssH * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const b = bounds(series);
    const W = cssW - PAD.l - PAD.r, H = cssH - PAD.t - PAD.b;
    const sx = x => PAD.l + ((x - b.xmin) / (b.xmax - b.xmin)) * W;
    const sy = y => PAD.t + (1 - (y - b.ymin) / (b.ymax - b.ymin)) * H;

    ctx.font = "11px -apple-system, sans-serif";
    ctx.strokeStyle = "#262b36"; ctx.fillStyle = "#8b93a7"; ctx.lineWidth = 1;
    // y gridlines + labels
    for (let i = 0; i <= 4; i++) {
      const v = b.ymin + (i / 4) * (b.ymax - b.ymin), y = sy(v);
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(cssW - PAD.r, y); ctx.stroke();
      ctx.fillText(fmt(v), 4, y + 3);
    }
    // x labels (min/mid/max)
    for (const t of [0, 0.5, 1]) {
      const v = b.xmin + t * (b.xmax - b.xmin);
      ctx.fillText(Math.round(v).toString(), sx(v) - 6, cssH - 6);
    }
    // lines
    for (const s of series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.6; ctx.beginPath();
      let started = false;
      for (const [x, y] of s.points) {
        if (x == null || y == null) continue;
        const px = sx(x), py = sy(y);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    // Native-title hover: show the nearest point's value as you move the mouse.
    canvas.onmousemove = (e) => {
      const mx = e.clientX - canvas.getBoundingClientRect().left;
      let best = null, bestDist = Infinity;
      for (const s of series) for (const [x, y] of s.points) {
        if (x == null || y == null) continue;
        const d = Math.abs(sx(x) - mx);
        if (d < bestDist) { bestDist = d; best = { x, y, label: s.label }; }
      }
      canvas.title = (best && bestDist <= 40) ? `${best.label}  step ${best.x}: ${fmt(best.y)}` : "";
    };
  }

  window.drawLineChart = drawLineChart;
})();
