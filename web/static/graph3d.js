/* Rotating 3D knowledge graph on an SVG, per the handoff spec:
   nodes on a sphere (Fibonacci distribution), rotation about Y at 0.18 rad/s,
   perspective scale 1 + z*0.35, depth opacity 0.3 + (z+1)*0.35, edges 0.8px
   #1d1d1f at opacity 0.1 + (z+1)*0.16, per-status node styling, depth sort
   every frame. Entities are nodes; facts are the edges joining them. */

const TAU = Math.PI * 2;

export function createGraph(svg) {
  let nodes = [];          // {id,label,status,x,y,z, degree}
  let edges = [];          // {source,target,status}
  let radiusFactor = 0.34; // of min(view w,h)
  let angle = 0;
  let last = performance.now();
  let raf = null;
  let rotate = true;

  function layout() {
    // Fibonacci sphere: even distribution for any node count.
    const n = nodes.length;
    const golden = Math.PI * (3 - Math.sqrt(5));
    nodes.forEach((node, i) => {
      const y = n === 1 ? 0 : 1 - (i / (n - 1)) * 2;
      const r = Math.sqrt(Math.max(0, 1 - y * y));
      const th = golden * i;
      node.x = Math.cos(th) * r;
      node.y = y;
      node.z = Math.sin(th) * r;
    });
  }

  function nodeR(node) {
    return 4 + Math.min(6, (node.degree || 1) * 1.4);
  }

  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (rotate) angle = (angle + 0.18 * dt) % TAU;

    const W = svg.clientWidth, H = svg.clientHeight;
    const R = Math.min(W, H) * radiusFactor;
    const cx = W / 2, cy = H / 2;
    const cosA = Math.cos(angle), sinA = Math.sin(angle);
    const t = now / 1000;

    const proj = new Map();
    for (const node of nodes) {
      const x = node.x * cosA + node.z * sinA;
      const z = -node.x * sinA + node.z * cosA;
      const s = 1 + z * 0.35;
      proj.set(node.id, {
        px: cx + x * R * s, py: cy + node.y * R * s, z, s,
        op: 0.3 + (z + 1) * 0.35,
      });
    }

    let out = "";
    for (const e of edges) {
      const a = proj.get(e.source), b = proj.get(e.target);
      if (!a || !b) continue;
      const z = (a.z + b.z) / 2;
      const dash = e.status === "invalidated" ? ' stroke-dasharray="2 2"' : "";
      const col = e.status === "invalidated" ? "#9d9da5"
        : (e.status === "added" || e.status === "revised") ? "#0a66ff" : "#1d1d1f";
      out += `<line x1="${a.px.toFixed(1)}" y1="${a.py.toFixed(1)}" `
           + `x2="${b.px.toFixed(1)}" y2="${b.py.toFixed(1)}" stroke="${col}" `
           + `stroke-width="0.8" opacity="${(0.1 + (z + 1) * 0.16).toFixed(2)}"${dash}/>`;
    }

    const sorted = [...nodes].sort((a, b) => proj.get(a.id).z - proj.get(b.id).z);
    for (const node of sorted) {
      const p = proj.get(node.id);
      let r = nodeR(node) * p.s;
      let circle;
      const st = node.status || "confirmed";
      if (st === "added") {
        r *= 1 + 0.12 * Math.sin(t * TAU / 2.8);            // ±12% pulse
        circle = `fill="#0a66ff" stroke="none"`;
      } else if (st === "revised") {
        circle = `fill="#fff" stroke="#0a66ff" stroke-width="1.6"`;
      } else if (st === "invalidated") {
        circle = `fill="#fff" stroke="#9d9da5" stroke-width="1" stroke-dasharray="2 2"`;
      } else {
        circle = `fill="#fff" stroke="#5b5b63" stroke-width="1"`;
      }
      const fs = (9.5 + 3.5 * Math.max(0, p.z)).toFixed(1);
      const deco = st === "invalidated"
        ? ' text-decoration="line-through" fill="#9d9da5"' : ' fill="#1d1d1f"';
      const label = (node.label || "").slice(0, 26);
      out += `<circle cx="${p.px.toFixed(1)}" cy="${p.py.toFixed(1)}" `
           + `r="${r.toFixed(1)}" ${circle} opacity="${p.op.toFixed(2)}"/>`
           + `<text x="${(p.px + r + 5).toFixed(1)}" y="${(p.py + 3).toFixed(1)}" `
           + `font-size="${fs}"${deco} opacity="${p.op.toFixed(2)}" `
           + `font-family="Geist, system-ui, sans-serif">${esc(label)}</text>`;
    }
    svg.innerHTML = out;
    raf = requestAnimationFrame(frame);
  }

  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  raf = requestAnimationFrame(frame);
  document.addEventListener("visibilitychange", () => {
    rotate = !document.hidden;
  });

  return {
    update(newNodes, newEdges, nodeStatus = {}, edgeStatus = {}) {
      const degree = {};
      for (const e of newEdges) {
        degree[e.source] = (degree[e.source] || 0) + 1;
        degree[e.target] = (degree[e.target] || 0) + 1;
      }
      nodes = newNodes.map(n => ({
        ...n, status: nodeStatus[n.id] || "confirmed", degree: degree[n.id] || 1,
      }));
      edges = newEdges.map(e => ({
        source: e.source, target: e.target,
        status: edgeStatus[e.id] ||
                (e.invalid_at ? "invalidated" : "confirmed"),
      }));
      layout();
    },
    zoom(dz) { radiusFactor = Math.max(0.15, Math.min(0.6, radiusFactor + dz)); },
    setRotate(on) { rotate = on; },
    destroy() { cancelAnimationFrame(raf); },
  };
}
