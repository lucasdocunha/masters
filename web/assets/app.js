// Inspeção CEAP — SPA estática, sem dependências no carregamento inicial.
// Cada seção baixa só o JSON que usa; Three.js é carregado apenas na página de espaço latente.

const PAGE = 50;
const cache = new Map();
const app = document.getElementById("app");

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => (n == null || Number.isNaN(n) ? "—" : Number(n).toLocaleString("pt-BR"));
const pct = (n) => (n == null ? "—" : `${Number(n).toFixed(1)}%`);
const bytes = (n) => {
  if (n == null) return "—";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};

async function load(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${path}`);
      return path.endsWith(".gz") ? r.json() : r.json();
    }));
  }
  return cache.get(path);
}
const loadText = (path) => {
  const key = "txt:" + path;
  if (!cache.has(key)) cache.set(key, fetch(path).then((r) => { if (!r.ok) throw new Error(`${r.status} ${path}`); return r.text(); }));
  return cache.get(key);
};

const state = { manifest: null, envs: null, models: null, logs: null, dataset: null };
const getManifest = async () => (state.manifest ??= await load("data/manifest.json"));
const getEnvs = async () => (state.envs ??= (await load("data/envs.json")).envs);
const getModels = async () => (state.models ??= await load("data/models.json"));
const getLogs = async () => (state.logs ??= (await load("data/logs.json")).logs);
// dataset gz: o navegador descomprime nativamente com DecompressionStream
async function getDataset() {
  if (!state.dataset) {
    state.dataset = (async () => {
      const r = await fetch("data/plotly_dataset.json.gz");
      if (!r.ok) throw new Error("dataset indisponível");
      const stream = r.body.pipeThrough(new DecompressionStream("gzip"));
      return JSON.parse(await new Response(stream).text());
    })();
  }
  return state.dataset;
}

function setTitle(t) { document.title = `${t} · CEAP`; }
function markNav() {
  const h = location.hash.replace(/^#\/?/, "").split("/")[0] || "";
  document.querySelectorAll(".side a").forEach((a) => {
    const target = a.getAttribute("href").replace(/^#\/?/, "").split("/")[0] || "";
    a.classList.toggle("on", target === h);
    if (target === h) a.scrollIntoView({ block: "nearest", inline: "nearest" });
  });
}

function pager(total, page, base) {
  const pages = Math.max(1, Math.ceil(total / PAGE));
  return `<div class="pager"><button class="btn" data-page="${page - 1}" ${page <= 0 ? "disabled" : ""}>‹ Anterior</button>
    <span>Página ${page + 1} de ${pages} · ${fmt(total)} itens</span>
    <button class="btn" data-page="${page + 1}" ${page >= pages - 1 ? "disabled" : ""}>Próxima ›</button></div>`;
}
function bindPager(root, onPage) {
  root.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => onPage(Number(b.dataset.page))));
}

// ---------------------------------------------------------------- Markdown mínimo (títulos, tabelas, listas, negrito, código)
function md(src) {
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const out = [];
  const lines = src.split("\n");
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\|/.test(l) && /^\|[\s:|-]+\|?\s*$/.test(lines[i + 1] || "")) {
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const head = cells(l);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<div class="tbl-wrap"><table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join("")}</tr></thead><tbody>${
        rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
      continue;
    }
    const h = l.match(/^(#{1,4})\s+(.*)/);
    if (h) out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`);
    else if (/^\s*[-*]\s+/.test(l)) out.push(`<ul>${[]}<li>${inline(l.replace(/^\s*[-*]\s+/, ""))}</li></ul>`);
    else if (/^---+$/.test(l)) out.push("<hr>");
    else if (l.trim()) out.push(`<p>${inline(l)}</p>`);
    i++;
  }
  return `<div class="md">${out.join("\n").replace(/<\/ul>\n<ul>/g, "")}</div>`;
}

// ---------------------------------------------------------------- páginas
async function pageOverview() {
  setTitle("Visão geral");
  const m = await getManifest();
  const s = m.stats;
  const envs = await getEnvs();
  document.getElementById("build-info").textContent = `Gerado em ${new Date(m.generated_at).toLocaleString("pt-BR")}`;
  const stat = (v, l) => `<div class="card stat"><b>${v}</b><span>${l}</span></div>`;
  app.innerHTML = `
    <h1>Visão geral</h1><p class="sub">Clusterizações, documentos e rotulagem do dataset CEAP.</p>
    <a class="card feature" href="#/latente"><span><strong>Espaço latente</strong><span class="muted">Explore todos os documentos em uma nuvem 3D e abra cada um.</span></span><span class="btn btn--primary">Abrir espaço latente</span></a>
    <div class="grid">
      ${stat(fmt(s.environments), "ambientes de clusterização")}
      ${stat(fmt(s.clusters), "clusters no total")}
      ${stat(fmt(s.sampled_docs), "documentos amostrados")}
      ${stat(fmt(s.dataset_docs), "documentos no dataset de rotulagem")}
      ${stat(fmt(s.labeled_docs), "anotações (histórico)")}
      ${stat(fmt(s.artifacts), "arquivos catalogados")}
      ${stat(bytes(s.total_bytes), "volume dos artefatos")}
      ${stat(fmt(s.logs), "logs")}
    </div>
    <h2>Ambientes de clusterização</h2>
    <div class="grid">${envs.map((e) => `
      <a class="card" href="#/ambientes/${esc(e.id)}"><strong>${esc(e.label)}</strong>
        <div class="muted">K = ${e.k} · ${fmt(e.total_docs)} docs</div></a>`).join("")}</div>`;
}

async function pageEnvs() {
  setTitle("Ambientes");
  const envs = await getEnvs();
  app.innerHTML = `<h1>Ambientes de clusterização</h1>
    <p class="sub">Cada ambiente é uma clusterização do mesmo conjunto com um espaço de representação diferente.</p>
    <div class="tbl-wrap"><table><thead><tr><th>Ambiente</th><th class="num">K</th><th class="num">Documentos</th><th>Clusters</th></tr></thead><tbody>
    ${envs.map((e) => `<tr><td><a href="#/ambientes/${esc(e.id)}">${esc(e.label)}</a><div class="muted"><code>${esc(e.id)}</code></div></td>
      <td class="num">${e.k}</td><td class="num">${fmt(e.total_docs)}</td>
      <td style="min-width:260px">${e.clusters.map((c) => `<span class="chip" title="${esc(c.taxonomy || "")}">${esc(c.id)} · ${pct(c.percent)}</span>`).join(" ")}</td></tr>`).join("")}
    </tbody></table></div>`;
}

async function pageEnv(envId, cid) {
  const envs = await getEnvs();
  const env = envs.find((e) => e.id === envId);
  if (!env) return notFound();
  if (cid == null) {
    setTitle(env.label);
    const maxPct = Math.max(...env.clusters.map((c) => c.percent || 0), 1);
    app.innerHTML = `<div class="crumbs"><a href="#/ambientes">Ambientes</a> / ${esc(env.label)}</div>
      <h1>${esc(env.label)}</h1><p class="sub">K = ${env.k} · ${fmt(env.total_docs)} documentos · <code>${esc(env.id)}</code></p>
      <div class="grid wide">
      ${env.clusters.map((c) => `
        <a class="card" href="#/ambientes/${esc(env.id)}/${esc(c.id)}">
          <div class="card-head"><strong>Cluster ${esc(c.id)}</strong><span class="muted">${fmt(c.total)}</span></div>
          <div class="bar" style="margin:12px 0"><i style="width:${((c.percent || 0) / maxPct * 100).toFixed(1)}%"></i></div>
          <div class="muted">${pct(c.percent)}${c.taxonomy ? ` · ${esc(c.taxonomy)}` : ""}</div>
          ${c.dominant_subcota ? `<div class="muted small">${esc(c.dominant_subcota)} (${pct(c.dominant_pct)})</div>` : ""}
          <div class="thumbs">${c.samples.filter((s) => s.thumb).slice(0, 4)
            .map((s) => `<img loading="lazy" decoding="async" src="${esc(s.thumb)}" alt="${esc(s.doc_name)}">`).join("")}</div>
        </a>`).join("")}</div>`;
    return;
  }
  const c = env.clusters.find((x) => x.id === cid);
  if (!c) return notFound();
  setTitle(`Cluster ${cid} · ${env.label}`);
  app.innerHTML = `<div class="crumbs"><a href="#/ambientes">Ambientes</a> / <a href="#/ambientes/${esc(env.id)}">${esc(env.label)}</a> / Cluster ${esc(cid)}</div>
    <h1>Cluster ${esc(cid)}</h1>
    <p class="sub">${fmt(c.total)} documentos (${pct(c.percent)})${c.taxonomy ? ` · taxonomia inferida: <strong>${esc(c.taxonomy)}</strong>` : ""}</p>
    ${c.top_subcotas?.length ? `<h2>Subcotas dominantes</h2><div class="tbl-wrap"><table><thead><tr><th>Subcota</th><th class="num">Docs</th><th class="num">%</th></tr></thead><tbody>
      ${c.top_subcotas.map(([n, q, p]) => `<tr><td>${esc(n)}</td><td class="num">${fmt(q)}</td><td class="num">${pct(p)}</td></tr>`).join("")}</tbody></table></div>` : ""}
    <h2>Documentos amostrados (${c.samples.length})</h2>
    ${c.samples.length ? `<div class="tbl-wrap"><table><thead><tr><th>#</th><th>Documento</th><th>Ano</th><th class="num">Págs</th><th>Subcota / fornecedor</th><th class="num">Distância</th></tr></thead><tbody>
      ${c.samples.map((s) => `<tr><td>${s.rank ?? ""}</td>
        <td>${s.thumb ? `<img class="doc-img" loading="lazy" decoding="async" src="${esc(s.thumb)}" alt="">` : ""}<div><code>${esc(s.doc_name)}</code></div></td>
        <td>${esc(s.year ?? "")}</td><td class="num">${esc(s.pages ?? "")}</td>
        <td>${esc(s.subcota ?? "")}<div class="muted">${esc(s.fornecedor ?? "")}</div></td>
        <td class="num">${s.distance == null ? "—" : Number(s.distance).toFixed(4)}</td></tr>`).join("")}</tbody></table></div>`
      : `<div class="notice">Sem amostras exportadas para este cluster.</div>`}`;
}

async function pageDocs() {
  setTitle("Documentos");
  const ds = await getDataset();
  const classes = [...new Set(ds.docs.map((d) => d.class))].sort();
  let q = "", cls = "", page = 0;
  const filtered = () => {
    const needle = q.trim().toLowerCase();
    return ds.docs.filter((d) => (!cls || d.class === cls) &&
      (!needle || `${d.name} ${d.fornecedor} ${d.subcota} ${d.id}`.toLowerCase().includes(needle)));
  };
  const draw = () => {
    const rows = filtered();
    const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
    app.innerHTML = `<h1>Documentos</h1>
      <p class="sub">${fmt(ds.docs.length)} documentos do dataset de rotulagem · ${classes.length} classes.</p>
      <div class="toolbar"><input type="search" id="q" placeholder="Buscar por nome, fornecedor, subcota…" value="${esc(q)}">
        <select id="cls"><option value="">Todas as classes</option>${classes.map((c) => `<option ${c === cls ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></div>
      <div class="tbl-wrap"><table><thead><tr><th>Documento</th><th>Classe</th><th>Ano</th><th>Fornecedor</th><th>Subcota</th><th>Status</th></tr></thead><tbody>
      ${slice.map((d) => `<tr><td><code>${esc(d.name)}</code></td><td><span class="chip">${esc(d.class)}</span></td><td>${esc(d.ano ?? "")}</td>
        <td>${esc(d.fornecedor ?? "")}</td><td class="muted">${esc(d.subcota ?? "")}</td><td class="muted">${esc(d.status ?? "")}</td></tr>`).join("")}
      </tbody></table></div>${pager(rows.length, page)}`;
    app.querySelector("#q").addEventListener("input", (e) => { q = e.target.value; page = 0; draw(); app.querySelector("#q").focus(); });
    app.querySelector("#cls").addEventListener("change", (e) => { cls = e.target.value; page = 0; draw(); });
    bindPager(app, (p) => { page = p; draw(); });
  };
  draw();
}

let leaveLatent = null;
async function pageLatent() {
  setTitle("Espaço latente");
  app.classList.add("is-latent");
  const [{ mountLatent }, index] = await Promise.all([import("./latent.js"), load("data/latent/index.json")]);
  if (!location.hash.startsWith("#/latente")) return;
  leaveLatent = mountLatent(app, index, getDataset);
}

async function pageLabeling() {
  setTitle("Tarefas");
  const [m, ds, latent] = await Promise.all([getModels(), getDataset(), load("data/latent/index.json")]);
  const hexOf = Object.fromEntries(latent.classes.map((c) => [c.name, c.hex]));
  const total = m.annotations.total || 1;
  const rows = Object.entries(m.annotations.by_class);
  const max = Math.max(1, ...rows.map(([, v]) => v));
  const counts = Object.values(ds.palette).filter((p) => p.name !== "Não Mapeado").sort((a, b) => b.count - a.count);
  const status = {};
  ds.docs.forEach((d) => { status[d.status ?? "—"] = (status[d.status ?? "—"] || 0) + 1; });
  app.innerHTML = `<h1>Tarefas de rotulagem</h1>
    <p class="sub">${fmt(total)} anotações registradas no histórico de ML · ${fmt(ds.docs.length)} documentos no dataset de rotulagem.</p>
    <div class="grid">${Object.entries(status).map(([k, v]) => `<div class="card stat"><b>${fmt(v)}</b><span>status: ${esc(k)}</span></div>`).join("")}</div>
    <h2>Anotações por classe (histórico)</h2>
    <div class="tbl-wrap"><table><thead><tr><th>Classe</th><th class="num">Anotações</th><th class="num">%</th><th style="width:40%"></th></tr></thead><tbody>
    ${rows.map(([c, v]) => `<tr><td><span class="chip">${esc(c)}</span></td><td class="num">${fmt(v)}</td><td class="num">${pct((v / total) * 100)}</td>
      <td><div class="bar"><i style="width:${(v / max * 100).toFixed(1)}%"></i></div></td></tr>`).join("")}</tbody></table></div>
    <h2>Documentos por classe (dataset de rotulagem)</h2>
    <div class="tbl-wrap"><table><thead><tr><th>Classe</th><th class="num">Documentos</th></tr></thead><tbody>
    ${counts.map((p) => `<tr><td><span class="dot" style="background:${esc(hexOf[p.name] ?? p.hex)}"></span>${esc(p.name)}</td><td class="num">${fmt(p.count)}</td></tr>`).join("")}</tbody></table></div>`;
}

async function pageResults() {
  setTitle("Resultados");
  const files = [
    ["Comparativo de pipelines", "files/evaluation/pdfs_selecionados/relatorio_comparativo_pipelines.md"],
    ["Avaliação da extração (Gemini)", "files/evaluation/pdfs_selecionados/relatorio_avaliacao.md"],
  ];
  const parts = await Promise.all(files.map(async ([t, p]) => {
    try { return `<h2>${esc(t)}</h2>${md(await loadText(p))}`; }
    catch { return `<h2>${esc(t)}</h2><div class="notice">Relatório não publicado.</div>`; }
  }));
  app.innerHTML = `<h1>Resultados</h1><p class="sub">Relatórios gerados pelos scripts de avaliação.</p>${parts.join("")}`;
}

async function pageModels() {
  setTitle("Modelos");
  const [m, ds] = await Promise.all([getModels(), getDataset()]);
  app.innerHTML = `<h1>Modelos testados</h1><p class="sub">Espaços de representação comparados e artefatos treinados.</p>
    <div class="grid wide">
    ${Object.values(ds.models).map((x) => `<div class="card"><strong>${esc(x.name)}</strong>
      <div class="muted">${esc(x.badge || "")} · ${esc(x.type || "")}</div>
      <p class="muted small">${esc(x.description || "")}</p>
      <div class="muted small">inter: ${esc(x.inter_dist ?? "—")} · intra: ${esc(x.intra_sim ?? "—")}</div></div>`).join("")}</div>
    <h2>Artefatos de modelo</h2>
    <div class="tbl-wrap"><table><thead><tr><th>Arquivo</th><th>Tipo</th><th class="num">Tamanho</th></tr></thead><tbody>
    ${m.artifacts.map((a) => `<tr><td><code>${esc(a.name)}</code></td><td>${esc(a.kind)}</td><td class="num">${bytes(a.size)}</td></tr>`).join("")}</tbody></table></div>`;
}

async function pageLogs() {
  setTitle("Logs");
  const logs = await getLogs();
  app.innerHTML = `<h1>Logs</h1><p class="sub">Últimas 200 linhas de cada log encontrado (slurm-*.out, *.log, logs/).</p>
    ${logs.length ? logs.map((l) => `<h2><code>${esc(l.path)}</code> <span class="muted">${fmt(l.lines)} linhas · ${bytes(l.size)}</span></h2><pre>${esc(l.tail.join("\n"))}</pre>`).join("")
      : `<div class="notice">Nenhum log encontrado. Os jobs do cluster (Slurm) gravam em <code>slurm-*.out</code> na raiz; rode <code>uv run python scripts/web/build_site.py</code> depois de executá-los.</div>`}`;
}

const KIND_FILTERS = ["Todos", "JSON", "CSV", "Markdown", "HTML", "Modelo", "Vetores", "Log", "Imagem", "Outro"];
async function pageFiles() {
  setTitle("Arquivos");
  const m = await getManifest();
  let kind = "Todos", q = "", page = 0;
  const draw = () => {
    const needle = q.trim().toLowerCase();
    const rows = m.artifacts.filter((a) => (kind === "Todos" || a.kind === kind) && (!needle || a.rel.toLowerCase().includes(needle)));
    const slice = rows.slice(page * PAGE, page * PAGE + PAGE);
    app.innerHTML = `<h1>Arquivos</h1><p class="sub">${fmt(m.artifacts.length)} arquivos catalogados. Os de texto pequenos podem ser visualizados; os demais aparecem só com tamanho.</p>
      <div class="toolbar"><input type="search" id="q" placeholder="Filtrar pelo caminho…" value="${esc(q)}">
      <select id="kind">${KIND_FILTERS.map((k) => `<option ${k === kind ? "selected" : ""}>${k}</option>`).join("")}</select></div>
      <div class="tbl-wrap"><table><thead><tr><th>Caminho</th><th>Tipo</th><th>Categoria</th><th class="num">Tamanho</th><th>Modificado</th></tr></thead><tbody>
      ${slice.map((a) => `<tr><td>${a.preview ? `<a href="#" data-prev="${esc(a.rel)}"><code>${esc(a.rel)}</code></a>` : `<code>${esc(a.rel)}</code>`}</td>
        <td><span class="chip">${esc(a.kind)}</span></td><td class="muted">${esc(a.category)}</td><td class="num">${bytes(a.size)}</td><td class="muted">${esc(a.modified)}</td></tr>`).join("")}
      </tbody></table></div>${pager(rows.length, page)}`;
    app.querySelector("#q").addEventListener("input", (e) => { q = e.target.value; page = 0; draw(); app.querySelector("#q").focus(); });
    app.querySelector("#kind").addEventListener("change", (e) => { kind = e.target.value; page = 0; draw(); });
    app.querySelectorAll("[data-prev]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openPreview(a.dataset.prev, m.artifacts.find((x) => x.rel === a.dataset.prev)); }));
    bindPager(app, (p) => { page = p; draw(); });
  };
  draw();
}

// Pré-visualização: JSON formatado, CSV em tabela (200 linhas), Markdown renderizado, demais como texto.
async function openPreview(rel, art) {
  const modal = document.getElementById("modal");
  document.getElementById("modal-title").textContent = rel;
  const body = document.getElementById("modal-body");
  body.innerHTML = `<p class="muted">Carregando…</p>`;
  modal.hidden = false;
  try {
    const text = await loadText(art.preview);
    if (art.kind === "JSON") {
      let pretty = text;
      try { pretty = JSON.stringify(JSON.parse(text), null, 2); } catch {}
      body.innerHTML = `<pre>${esc(pretty.slice(0, 200_000))}</pre>`;
    } else if (art.kind === "CSV") {
      const lines = text.split(/\r?\n/).filter(Boolean).slice(0, 201);
      const sep = lines[0]?.includes(";") ? ";" : ",";
      const rows = lines.map((l) => l.split(sep));
      body.innerHTML = `<div class="tbl-wrap"><table><thead><tr>${rows[0].map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>
        ${rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    } else if (art.kind === "Markdown") {
      body.innerHTML = md(text);
    } else if (art.kind === "HTML") {
      body.innerHTML = `<iframe src="${esc(art.preview)}" style="width:100%;height:70vh;border:0"></iframe>`;
    } else {
      body.innerHTML = `<pre>${esc(text.slice(0, 200_000))}</pre>`;
    }
  } catch (err) {
    body.innerHTML = `<div class="notice">${esc(err.message)}</div>`;
  }
}
document.getElementById("modal-close").addEventListener("click", () => (document.getElementById("modal").hidden = true));
document.getElementById("modal").addEventListener("click", (e) => { if (e.target.id === "modal") e.target.hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") document.getElementById("modal").hidden = true; });

function notFound() { app.innerHTML = `<h1>Não encontrado</h1><p class="sub"><a href="#/">Voltar à visão geral</a></p>`; }

// ---------------------------------------------------------------- roteador
const ROUTES = [
  [/^$/, () => pageOverview()],
  [/^ambientes$/, () => pageEnvs()],
  [/^ambientes\/([^/]+)$/, (m) => pageEnv(decodeURIComponent(m[1]), null)],
  [/^ambientes\/([^/]+)\/([^/]+)$/, (m) => pageEnv(decodeURIComponent(m[1]), decodeURIComponent(m[2]))],
  [/^documentos$/, () => pageDocs()],
  [/^latente$/, () => pageLatent()],
  [/^rotulagem$/, () => pageLabeling()],
  [/^resultados$/, () => pageResults()],
  [/^modelos$/, () => pageModels()],
  [/^logs$/, () => pageLogs()],
  [/^arquivos$/, () => pageFiles()],
];

async function route() {
  let path = location.hash.replace(/^#\/?/, "").replace(/\/+$/, "");
  if (!path && location.pathname && location.pathname !== "/") {
    path = location.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
  }
  leaveLatent?.();
  leaveLatent = null;
  app.classList.remove("is-latent");
  markNav();
  window.scrollTo({ top: 0 });
  for (const [re, fn] of ROUTES) {
    const m = path.match(re);
    if (m) {
      try { await fn(m); }
      catch (err) { app.innerHTML = `<h1>Erro</h1><div class="notice">${esc(err.message)}</div>`; }
      return;
    }
  }
  notFound();
}
window.addEventListener("hashchange", route);
route();
