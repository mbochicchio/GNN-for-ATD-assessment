#!/usr/bin/env python3
"""
Interactive HTML view of one project of the GNN-ATD dataset
(output of build_dataset.py: nodes.csv, edges.csv, meta.json).

  python visualize_dataset.py data/dataset/java/DependencyCheck
  -> writes data/dataset/java/DependencyCheck/graph.html

The page is self-contained (data embedded) and loads only d3 from cdnjs.
Nodes = packages/namespaces, size ~ sqrt(LOC), colour = selected smell or
number of smells; edges = dependencies (arrow: dependant -> dependee).
"""
import csv
import json
import sys
from pathlib import Path

FEATURES = ["Ab", "In", "D", "FI", "FO", "LOC", "PR"]
LABELS = ["CD", "UD", "AI", "GC", "FC", "SF"]


def load(project_dir):
    d = Path(project_dir)
    with open(d / "nodes.csv", encoding="utf-8") as f:
        nodes = []
        for r in csv.DictReader(f):
            n = {"id": int(r["node_id"]), "name": r["name"],
                 "mask": int(r["label_mask"]), "atdi": float(r["atdi"]),
                 "units": int(float(r.get("n_units", 0) or 0))}
            n.update({k: float(r[k]) for k in FEATURES})
            n.update({k: int(r[k]) for k in LABELS})
            nodes.append(n)
    with open(d / "edges.csv", encoding="utf-8") as f:
        edges = [{"source": int(r["src"]), "target": int(r["dst"]), "w": float(r["weight"])}
                 for r in csv.DictReader(f)]
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return {"nodes": nodes, "edges": edges, "meta": meta}


def write_html(project_dir):
    """Write <project_dir>/graph.html from nodes.csv, edges.csv, meta.json."""
    d = Path(project_dir)
    data = load(d)
    html = TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
        "__TITLE__", data["meta"].get("project", d.name))
    out = d / "graph.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    print(f"written {write_html(sys.argv[1])}")
    return 0


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__ · dependency graph</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#EEF1F0; --panel:#F8FAF9; --ink:#1B2733; --muted:#5E6B78; --rule:#D3DAD8;
  --edge:#9AA7B2; --clean:#C3CCD3; --focus:#1B2733;
  --CD:#D55E00; --UD:#0072B2; --AI:#CC79A7; --GC:#009E73; --FC:#E69F00; --SF:#56B4E9;
  --s0:#C3CCD3; --s1:#F2C27B; --s2:#E58A3C; --s3:#C4462A; --s4:#8E1F2A;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#141B21; --panel:#1B242C; --ink:#E4EAEE; --muted:#93A1AD; --rule:#2C3842;
  --edge:#4E5E6B; --clean:#3E4B56; --focus:#E4EAEE; --s0:#3E4B56;
}}
:root[data-theme="dark"]{
  --bg:#141B21; --panel:#1B242C; --ink:#E4EAEE; --muted:#93A1AD; --rule:#2C3842;
  --edge:#4E5E6B; --clean:#3E4B56; --focus:#E4EAEE; --s0:#3E4B56;
}
*,*::before,*::after{box-sizing:inherit}
html{height:100%; scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0; height:100%; background:var(--bg); color:var(--ink);
  font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
.app{display:grid; grid-template-columns:320px 1fr; height:100%}
aside{background:var(--panel); border-right:1px solid var(--rule); overflow-y:auto; padding:20px 20px 28px}
h1{font:500 22px/1.2 "IBM Plex Sans Condensed","IBM Plex Sans",sans-serif; margin:0 0 4px; word-break:break-word}
.sub{color:var(--muted); margin:0 0 18px}
.stats{display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-bottom:18px}
.stat b{display:block; font-size:20px; font-weight:600}
.stat span{color:var(--muted); font-size:12px}
section{border-top:1px solid var(--rule); padding-top:14px; margin-top:14px}
h2{font-size:13px; font-weight:600; margin:0 0 8px}
label.ctl{display:block; color:var(--muted); font-size:12px; margin:8px 0 4px}
select,input[type=search]{width:100%; font:inherit; color:var(--ink); background:var(--bg);
  border:1px solid var(--rule); border-radius:6px; padding:6px 8px}
select:focus-visible,input:focus-visible,button:focus-visible{outline:2px solid var(--focus); outline-offset:2px}
.check{display:flex; gap:8px; align-items:center; margin-top:10px; color:var(--muted); font-size:13px}
.legend{display:flex; flex-wrap:wrap; gap:6px 12px; margin-top:10px; font-size:12px}
.dot{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; vertical-align:-1px}
.legend i{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px}
table{width:100%; border-collapse:collapse; font-size:13px}
td{padding:3px 0; border-bottom:1px solid var(--rule)}
td:last-child{text-align:right; font-variant-numeric:tabular-nums}
.chips{display:flex; flex-wrap:wrap; gap:6px; margin:8px 0}
.chip{font-size:12px; padding:2px 8px; border-radius:10px; color:#fff}
.chip.none{background:var(--clean); color:var(--ink)}
.detail-name{font-weight:600; word-break:break-all; margin-bottom:4px}
.note{color:var(--muted); font-size:12px}
ol.rank{margin:0; padding-left:20px; font-size:13px}
ol.rank li{margin:2px 0}
ol.rank button{all:unset; cursor:pointer; word-break:break-all}
ol.rank button:hover{text-decoration:underline}
ol.rank .v{color:var(--muted); font-variant-numeric:tabular-nums}
main{position:relative; overflow:hidden}
svg{width:100%; height:100%; display:block; cursor:grab}
.link{stroke:var(--edge); stroke-opacity:.55; fill:none}
.node circle{stroke:var(--panel); stroke-width:1.5}
.node.nomask circle{stroke-dasharray:3 2; stroke:var(--muted)}
.node text{font-size:11px; fill:var(--ink); pointer-events:none; paint-order:stroke;
  stroke:var(--bg); stroke-width:3px}
.dim{opacity:.12}
.node.sel circle{stroke:var(--focus); stroke-width:3}
#tip{position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--rule);
  border-radius:6px; padding:6px 9px; font-size:12px; max-width:320px; display:none; word-break:break-all}
.hint{position:absolute; right:14px; bottom:12px; color:var(--muted); font-size:12px}
@media (max-width:760px){
  .app{grid-template-columns:1fr; grid-template-rows:auto 70vh; height:auto}
  aside{border-right:0; border-bottom:1px solid var(--rule)}
}
@media (prefers-reduced-motion:reduce){ *{transition:none!important} }
</style>
</head>
<body>
<div class="app">
<aside>
  <h1 id="title"></h1>
  <p class="sub">Package-level dependency graph with Designite smell labels</p>
  <div class="stats">
    <div class="stat"><b id="nN"></b><span>components</span></div>
    <div class="stat"><b id="nE"></b><span>dependencies</span></div>
    <div class="stat"><b id="nS"></b><span>smell-affected components</span></div>
  </div>

  <label class="ctl" for="colorBy">Colour nodes by</label>
  <select id="colorBy">
    <option value="count">Number of smells</option>
    <option value="CD">Cyclic Dependency</option>
    <option value="UD">Unstable Dependency</option>
    <option value="AI">Ambiguous Interface</option>
    <option value="GC">God Component</option>
    <option value="FC">Feature Concentration</option>
    <option value="SF">Scattered Functionality</option>
    <option value="atdi">ATDI (Arcan)</option>
  </select>
  <div class="legend" id="legend"></div>

  <label class="ctl" for="search">Find a component</label>
  <input id="search" type="search" placeholder="e.g. analyzer" list="names" autocomplete="off">
  <datalist id="names"></datalist>
  <label class="check"><input type="checkbox" id="showLabels"> Show all node labels</label>

  <section id="detail">
    <h2>Selected component</h2>
    <p class="note">Select a node to inspect its features, labels, and dependencies.</p>
  </section>

  <section>
    <h2>Smell distribution</h2>
    <table id="smellTable"></table>
    <p class="note" id="ds"></p>
  </section>

  <section>
    <h2>Highest ATDI</h2>
    <ol class="rank" id="rank"></ol>
  </section>
</aside>
<main>
  <svg id="svg" role="img" aria-label="Dependency graph"></svg>
  <div id="tip"></div>
  <div class="hint">Drag to pan, scroll to zoom</div>
</main>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const DATA = __DATA__;
const LABELS = ["CD","UD","AI","GC","FC","SF"];
const NAMES = {CD:"Cyclic Dependency",UD:"Unstable Dependency",AI:"Ambiguous Interface",
  GC:"God Component",FC:"Feature Concentration",SF:"Scattered Functionality"};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const nodes = DATA.nodes.map(d => ({...d, nSmell: LABELS.reduce((s,k)=>s+d[k],0),
  short: d.name.split(".").slice(-1)[0] || d.name}));
const byId = new Map(nodes.map(d => [d.id, d]));
const links = DATA.edges.map(e => ({...e}));
const meta = DATA.meta;
let current = null;

// ---------- sidebar ----------
document.getElementById("title").textContent = meta.project;
document.getElementById("nN").textContent = nodes.length;
document.getElementById("nE").textContent = links.length;
document.getElementById("nS").textContent = nodes.filter(d => d.nSmell > 0).length;
document.getElementById("names").innerHTML = nodes.map(d => `<option value="${d.name}">`).join("");
document.getElementById("smellTable").innerHTML = LABELS.map(k =>
  `<tr><td><i class="dot" style="background:var(--${k})"></i>${NAMES[k]}</td>
   <td>${meta.label_counts[k]} / ${nodes.length}</td></tr>`).join("");
document.getElementById("ds").textContent = meta.dense_structure
  ? "Dense Structure detected at project level (not assigned to individual nodes)."
  : "Dense Structure not detected.";
const unmasked = nodes.filter(d => !d.mask).length;
if (unmasked) document.getElementById("ds").textContent +=
  ` ${unmasked} nodes not analysed by Designite (dashed outline): labels unavailable.`;
const rank = [...nodes].sort((a,b) => b.atdi - a.atdi).slice(0,10);
document.getElementById("rank").innerHTML = rank.map(d =>
  `<li><button data-id="${d.id}">${d.short}</button> <span class="v">${d.atdi.toFixed(1)}</span></li>`).join("");
document.querySelectorAll("#rank button").forEach(b =>
  b.addEventListener("click", () => select(byId.get(+b.dataset.id), true)));

// ---------- graph ----------
const svg = d3.select("#svg"), g = svg.append("g");
const W = () => svg.node().clientWidth, H = () => svg.node().clientHeight;
const maxLoc = d3.max(nodes, d => d.LOC) || 1;
const r = d3.scaleSqrt().domain([0, maxLoc]).range([4, 26]);
const wScale = d3.scaleLog().domain([1, d3.max(links, d => d.w) || 1]).range([0.6, 3.5]);
const maxAtdi = d3.max(nodes, d => d.atdi) || 1;

svg.append("defs").append("marker").attr("id","arrow").attr("viewBox","0 -4 8 8")
  .attr("refX",8).attr("refY",0).attr("markerUnits","userSpaceOnUse").attr("markerWidth",9).attr("markerHeight",9).attr("orient","auto")
  .append("path").attr("d","M0,-4L8,0L0,4").attr("fill","var(--edge)");

const link = g.append("g").selectAll("path").data(links).join("path")
  .attr("class","link").attr("stroke-width", d => wScale(Math.max(1,d.w))).attr("marker-end","url(#arrow)");
const node = g.append("g").selectAll("g").data(nodes).join("g")
  .attr("class", d => "node" + (d.mask ? "" : " nomask")).attr("tabindex",0)
  .attr("aria-label", d => d.name);
node.append("circle").attr("r", d => r(d.LOC));
node.append("text").attr("dy", d => -r(d.LOC) - 4).attr("text-anchor","middle").text(d => d.short);

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(95).strength(0.2))
  .force("charge", d3.forceManyBody().strength(-420))
  .force("collide", d3.forceCollide(d => r(d.LOC) + 6))
  .force("x", d3.forceX(() => W()/2).strength(0.05))
  .force("y", d3.forceY(() => H()/2).strength(0.05))
  .on("tick", ticked).on("end", fit);
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { sim.stop(); for (let i=0;i<300;i++) sim.tick(); ticked(); setTimeout(fit, 0); }
let fitted = false;
function fit(){
  if (fitted) return; fitted = true;
  const pad = 40, xs = nodes.map(d => d.x), ys = nodes.map(d => d.y);
  const x0 = d3.min(xs) - pad, x1 = d3.max(xs) + pad, y0 = d3.min(ys) - pad, y1 = d3.max(ys) + pad;
  const k = Math.min(2, 0.95 / Math.max((x1 - x0) / W(), (y1 - y0) / H()));
  svg.transition().duration(600).call(zoom.transform,
    d3.zoomIdentity.translate(W()/2, H()/2).scale(k).translate(-(x0 + x1)/2, -(y0 + y1)/2));
}

function ticked(){
  link.attr("d", d => {
    const dx = d.target.x - d.source.x, dy = d.target.y - d.source.y, L = Math.hypot(dx,dy) || 1;
    const rt = r(d.target.LOC) + 3, rs = r(d.source.LOC);
    return `M${d.source.x + dx*rs/L},${d.source.y + dy*rs/L}L${d.target.x - dx*rt/L},${d.target.y - dy*rt/L}`;
  });
  node.attr("transform", d => `translate(${d.x},${d.y})`);
}

node.call(d3.drag()
  .on("start", (e,d) => { if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
  .on("drag", (e,d) => { d.fx=e.x; d.fy=e.y; })
  .on("end", (e,d) => { if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));
const zoom = d3.zoom().scaleExtent([0.2,6]).on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom).on("click", e => { if (e.target === svg.node()) select(null); });

// ---------- colour ----------
function colour(d){
  const m = document.getElementById("colorBy").value;
  if (m === "count") return css("--s" + Math.min(d.nSmell, 4));
  if (m === "atdi") return d3.interpolateLab(css("--s0"), css("--s4"))(Math.sqrt(d.atdi / maxAtdi));
  return d[m] ? css("--" + m) : css("--clean");
}
function legend(){
  const m = document.getElementById("colorBy").value, el = document.getElementById("legend");
  if (m === "count") el.innerHTML = [0,1,2,3,4].map(i =>
    `<span><i style="background:var(--s${i})"></i>${i===4?"4+":i} ${i===1?"smell":"smells"}</span>`).join("");
  else if (m === "atdi") el.innerHTML = `<span><i style="background:var(--s0)"></i>0</span>
    <span><i style="background:var(--s4)"></i>${maxAtdi.toFixed(0)}</span>`;
  else el.innerHTML = `<span><i style="background:var(--${m})"></i>affected</span>
    <span><i style="background:var(--clean)"></i>not affected</span>`;
  el.innerHTML += `<span>node size ∝ LOC</span>`;
}
function paint(){ node.select("circle").attr("fill", colour); legend(); }
document.getElementById("colorBy").addEventListener("change", paint);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
paint();

// ---------- labels ----------
function labelsVisible(){
  const all = document.getElementById("showLabels").checked;
  node.select("text").style("display", d => all || r(d.LOC) > 12 || d === current ? null : "none");
}
document.getElementById("showLabels").addEventListener("change", labelsVisible);

// ---------- tooltip ----------
const tip = document.getElementById("tip");
node.on("mouseenter", (e,d) => {
  const s = LABELS.filter(k => d[k]).join(", ") || "none";
  tip.innerHTML = `<b>${d.name}</b><br>LOC ${d.LOC} · FI ${d.FI} · FO ${d.FO}<br>Smells: ${s}`;
  tip.style.display = "block";
}).on("mousemove", e => {
  const b = document.querySelector("main").getBoundingClientRect();
  tip.style.left = (e.clientX - b.left + 14) + "px"; tip.style.top = (e.clientY - b.top + 14) + "px";
}).on("mouseleave", () => tip.style.display = "none")
  .on("click", (e,d) => { e.stopPropagation(); select(d); })
  .on("keydown", (e,d) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(d); } });

// ---------- selection ----------
function select(d, center){
  current = d;
  node.classed("sel", n => n === d);
  if (!d){ node.classed("dim", false); link.classed("dim", false); renderDetail(null); labelsVisible(); return; }
  const out = links.filter(l => l.source === d), inc = links.filter(l => l.target === d);
  const nb = new Set([d, ...out.map(l => l.target), ...inc.map(l => l.source)]);
  node.classed("dim", n => !nb.has(n));
  link.classed("dim", l => l.source !== d && l.target !== d);
  renderDetail(d, out, inc); labelsVisible();
  if (center) svg.transition().duration(500).call(zoom.translateTo, d.x, d.y);
}
function renderDetail(d, out, inc){
  const el = document.getElementById("detail");
  if (!d){ el.innerHTML = `<h2>Selected component</h2><p class="note">Select a node to inspect its features, labels, and dependencies.</p>`; return; }
  const chips = LABELS.filter(k => d[k]).map(k => `<span class="chip" style="background:var(--${k})">${NAMES[k]}</span>`).join("")
    || `<span class="chip none">${d.mask ? "No smells" : "Labels unavailable"}</span>`;
  const f = (v, p=3) => Number.isInteger(v) ? v : v.toFixed(p);
  const rows = [["Abstractness (Ab)",d.Ab],["Instability (In)",d.In],["Distance (D)",d.D],["Fan-in (FI)",d.FI],
    ["Fan-out (FO)",d.FO],["Lines of code (LOC)",d.LOC],["PageRank (PR)",d.PR],["ATDI (Arcan)",d.atdi],["Classes",d.units]];
  el.innerHTML = `<h2>Selected component</h2><div class="detail-name">${d.name}</div>
    <div class="chips">${chips}</div>
    <table>${rows.map(([k,v]) => `<tr><td>${k}</td><td>${f(v)}</td></tr>`).join("")}</table>
    <p class="note">Efferent dependencies: ${out.length}. Afferent dependencies: ${inc.length}.</p>`;
}
document.getElementById("search").addEventListener("change", e => {
  const d = nodes.find(n => n.name === e.target.value) || nodes.find(n => n.name.includes(e.target.value));
  if (d) select(d, true);
});
labelsVisible();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    sys.exit(main())
