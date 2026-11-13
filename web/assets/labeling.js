// Rotulagem pelo site: modo documento (um PDF por vez, estilo Label Studio) e modo pasta (grade dos
// documentos mais próximos do centróide de uma classe). Tudo é gravado no git por api/labels.
const PDFJS = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38";
const PASSWORD_KEY = "ceap-editor-password";
const PENDING_KEY = "ceap-rotulagem-pendentes";
const REJECTED_KEY = "ceap-pasta-rejeitados";
const UNMAPPED = "_unmapped";
const AUTO_FLUSH = 25;
const PAGE_SIZES = [60, 120, 240, 480];
const PAGE_SIZE_KEY = "ceap-pasta-por-pagina";
const KEYS = "1234567890qwertyuiop";
const CHANNEL = "ceap-rotulos"; // mesmo canal do espaço latente: avisa as outras abas quando algo é salvo

// Avisa as outras abas e repassa avisos recebidos para quem estiver montado.
function openChannel(onEvents) {
  if (!("BroadcastChannel" in window)) return { post() {}, close() {} };
  const channel = new BroadcastChannel(CHANNEL);
  channel.onmessage = (event) => onEvents(event.data?.events ?? []);
  return { post: (events) => channel.postMessage({ events }), close: () => channel.close() };
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("pt-BR");
const sim = (n) => Number(n).toFixed(3).replace(".", ",");
const safeUrl = (url) => (/^https:\/\/www\.camara\.leg\.br\//.test(url ?? "") ? url : null);
const store = (kind) => (kind === "session" ? sessionStorage : localStorage);
const readJson = (key, fallback, kind) => { try { return JSON.parse(store(kind).getItem(key)) ?? fallback; } catch { return fallback; } };
const writeJson = (key, value, kind) => { try { store(kind).setItem(key, JSON.stringify(value)); } catch { /* sem armazenamento */ } };
const readPassword = () => { try { return sessionStorage.getItem(PASSWORD_KEY); } catch { return null; } };
const writePassword = (v) => { try { if (v) sessionStorage.setItem(PASSWORD_KEY, v); else sessionStorage.removeItem(PASSWORD_KEY); } catch { /* idem */ } };

// ---------------------------------------------------------------- dados
async function fetchJsonGz(path, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status} ao baixar ${path}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(bytes));
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
}

const queues = new Map();
function loadQueue(name, signal) {
  if (!queues.has(name)) {
    queues.set(name, fetchJsonGz(`data/review/${encodeURIComponent(name)}.json.gz`, signal).then((q) => {
      const rows = (key) => q[key].map((r) => Object.fromEntries(q.campos.map((field, i) => [field, r[i]])));
      return { ...q, candidatos: rows("candidatos"), auditoria: rows("auditoria") };
    }).catch((err) => { queues.delete(name); throw err; }));
  }
  return queues.get(name);
}

// Classe vigente de cada documento rotulado pelo site (último evento do log no git).
async function loadSaved(signal) {
  const saved = new Map();
  try {
    const response = await fetch("api/labels", { signal, cache: "no-store" });
    if (response.ok) for (const e of (await response.json()).corrections) saved.set(e.id, e.class);
  } catch { /* sem API: segue só com as filas */ }
  return saved;
}

// ---------------------------------------------------------------- pendentes e envio
const pending = new Map(readJson(PENDING_KEY, []).map((c) => [c.id, c]));
const persistPending = () => writeJson(PENDING_KEY, [...pending.values()]);
const changeFor = (doc, name, mode) => ({
  id: String(doc.id), class: name, point: Number.isInteger(doc.ponto) ? doc.ponto : -1, mode,
  meta: { fornecedor: doc.fornecedor ?? "", subcota: doc.subcota ?? "", ano: Number(doc.ano) || undefined, url: safeUrl(doc.url) ?? undefined },
});

async function send(changes, signal) {
  const password = readPassword();
  if (!password) return { error: "Informe a senha de edição.", needsPassword: true };
  let response;
  try {
    response = await fetch("api/labels", {
      method: "POST", signal, headers: { "Content-Type": "application/json" }, body: JSON.stringify({ password, changes }),
    });
  } catch {
    return { error: "Sem conexão com o serviço de rótulos." };
  }
  const body = await response.json().catch(() => ({}));
  if (response.status === 401) { writePassword(null); return { error: body.error ?? "Senha incorreta.", needsPassword: true }; }
  if (!response.ok) return { error: body.error ?? `Erro ${response.status}.` };
  return { ok: true, commit: body.commit, applied: body.applied ?? [] };
}

function passwordForm(onSaved) {
  const form = document.createElement("form");
  form.className = "rot-pass";
  form.innerHTML = `<label for="rot-pass">Senha de edição</label>
    <input type="password" id="rot-pass" autocomplete="current-password" required>
    <button class="btn" type="submit">Usar senha</button>`;
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    writePassword(form.querySelector("input").value);
    onSaved();
  });
  return form;
}

let pdfjs = null;
function loadPdfjs() {
  pdfjs ??= import(/* @vite-ignore */ `${PDFJS}/pdf.min.mjs`).then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = `${PDFJS}/pdf.worker.min.mjs`;
    return lib;
  }).catch((err) => { pdfjs = null; throw err; });
  return pdfjs;
}

// Primeira página como imagem, no máximo 4 em paralelo.
const thumbs = new Map();
let running = 0;
const waiting = [];
function thumbnail(url, width) {
  if (!thumbs.has(url)) {
    thumbs.set(url, new Promise((resolve, reject) => {
      waiting.push(async () => {
        running++;
        try {
          const lib = await loadPdfjs();
          const doc = await lib.getDocument({ url: `api/pdf?url=${encodeURIComponent(url)}`, disableAutoFetch: true }).promise;
          const page = await doc.getPage(1);
          const base = page.getViewport({ scale: 1 });
          const viewport = page.getViewport({ scale: (width * devicePixelRatio) / base.width });
          const canvas = document.createElement("canvas");
          canvas.width = Math.round(viewport.width);
          canvas.height = Math.round(viewport.height);
          await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
          doc.destroy();
          resolve(await new Promise((ok) => canvas.toBlob((b) => ok(URL.createObjectURL(b)), "image/webp", 0.8)));
        } catch (err) {
          thumbs.delete(url);
          reject(err);
        } finally {
          running--;
          waiting.shift()?.();
        }
      });
      if (running < 6) waiting.shift()?.();
    }));
  }
  return thumbs.get(url);
}

function shell(index, active, cls, extra = "") {
  return `<div class="rot-head">
      <div><h1>${active === "documento" ? "Rotular documentos" : "Pasta por classe"}</h1>
        <p class="sub">${active === "documento"
          ? "Um documento por vez. Escolha a classe pelo botão ou atalho; as decisões vão para o git em lotes."
          : "Documentos mais próximos do centróide da classe. Clique para tirar os que não são dela e confirme o resto."}</p></div>
      <nav class="segmented rot-modes" aria-label="Modo">
        <a href="#/rotular/${esc(cls)}" aria-current="${active === "documento"}">Documento</a>
        <a href="#/pasta/${esc(cls)}" aria-current="${active === "pasta"}">Pasta</a>
      </nav>
    </div>
    ${index.espaco?.endsWith("-3d") ? `<div class="notice rot-preview">Prévia: ranking calculado nas coordenadas 3D do site. Rode <code>scripts/labeling/build_review_queues.py</code> no cluster para usar os embeddings 128-D.</div>` : ""}
    ${extra}`;
}

const classOptions = (index, current) => index.classes
  .map((c) => `<option value="${esc(c.nome)}" ${c.nome === current ? "selected" : ""}>${esc(c.nome)} · ${fmt(c.rotulados)} rotulados</option>`).join("");

// ---------------------------------------------------------------- modo documento
export function mountDocMode(app, index, classes, cls, queueName) {
  const abort = new AbortController();
  const { signal } = abort;
  const queueKey = ["candidatos", "auditoria", "rejeitados"].includes(queueName) ? queueName : "candidatos";
  const options = [...classes.filter((c) => c !== UNMAPPED), UNMAPPED];
  let items = [], position = 0, saved = new Map(), message = "", busy = false;
  const channel = openChannel((events) => {
    for (const e of events) { saved.set(e.id, e.class); if (pending.get(e.id)?.class === e.class) pending.delete(e.id); }
    persistPending();
    if (items.length) draw();
  });

  app.innerHTML = shell(index, "documento", cls, `<div class="toolbar">
      <select id="rot-class" aria-label="Classe">${classOptions(index, cls)}</select>
      <select id="rot-queue" aria-label="Fila">
        <option value="candidatos" ${queueKey === "candidatos" ? "selected" : ""}>Candidatos (mais próximos do centróide)</option>
        <option value="auditoria" ${queueKey === "auditoria" ? "selected" : ""}>Auditoria (rotulados menos típicos)</option>
        <option value="rejeitados" ${queueKey === "rejeitados" ? "selected" : ""}>Tirados da pasta</option>
      </select>
      <input type="search" id="rot-find" placeholder="Ir para doc_id…" inputmode="numeric">
    </div>
    <div class="rot-doc" id="rot-doc"><p class="muted">Carregando fila…</p></div>`);
  const go = () => { location.hash = `#/rotular/${app.querySelector("#rot-class").value}/${app.querySelector("#rot-queue").value}`; };
  app.querySelector("#rot-class").addEventListener("change", go);
  app.querySelector("#rot-queue").addEventListener("change", go);
  app.querySelector("#rot-find").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const i = items.findIndex((d) => d.id === e.target.value.trim());
    message = i >= 0 ? "" : "Documento não está nesta fila.";
    if (i >= 0) position = i;
    draw();
  });
  const root = app.querySelector("#rot-doc");

  const decided = (doc) => pending.get(doc.id)?.class ?? saved.get(doc.id);
  const currentOf = (doc) => decided(doc) ?? (queueKey === "auditoria" ? cls : doc.atual || UNMAPPED);

  function draw() {
    if (!items.length) { root.innerHTML = `<div class="notice">Fila vazia.</div>`; return; }
    if (!root.querySelector(".rot-viewer")) root.innerHTML = `<div class="rot-viewer"></div><aside class="rot-side"></aside>`;
    position = Math.max(0, Math.min(position, items.length - 1));
    const doc = items[position];
    const current = currentOf(doc);
    const url = safeUrl(doc.url);
    const done = items.filter((d) => decided(d) !== undefined).length;
    const viewer = root.querySelector(".rot-viewer");
    if (viewer.dataset.id !== doc.id) {
      viewer.dataset.id = doc.id;
      viewer.innerHTML = url ? `<iframe src="${esc(url)}#view=FitH" title="PDF ${esc(doc.id)}"></iframe>` : `<div class="notice">Sem PDF.</div>`;
    }
    const side = root.querySelector(".rot-side");
    side.innerHTML = `<div class="rot-pos"><button class="btn" id="rot-prev" ${position === 0 ? "disabled" : ""} aria-label="Anterior (←)">←</button>
          <span class="mono">${fmt(position + 1)} / ${fmt(items.length)}</span>
          <button class="btn" id="rot-next" ${position >= items.length - 1 ? "disabled" : ""} aria-label="Próximo (→)">→</button>
          <span class="muted small">${fmt(done)} decididos</span></div>
        <h2 class="mono">${esc(doc.id)}</h2>
        <dl class="rot-meta">
          <dt>Classe</dt><dd><span class="chip ${pending.has(doc.id) ? "chip--pending" : ""}">${esc(current === UNMAPPED ? "sem rótulo" : current)}</span>${pending.has(doc.id) ? ` <span class="muted small">pendente</span>` : ""}</dd>
          <dt>Similaridade</dt><dd class="mono">${sim(doc.sim)} com ${esc(cls)}</dd>
          <dt>2ª classe</dt><dd>${esc(doc.segunda)} <span class="muted mono">(margem ${sim(doc.margem)})</span></dd>
          <dt>Fornecedor</dt><dd>${esc(doc.fornecedor || "—")}</dd>
          <dt>Subcota</dt><dd>${esc(doc.subcota || "—")}</dd>
          <dt>Ano</dt><dd>${esc(doc.ano || "—")}</dd>
        </dl>
        ${url ? `<a class="link" href="${esc(url)}" target="_blank" rel="noopener">Abrir PDF em outra aba</a>` : ""}
        <div class="rot-classes" role="group" aria-label="Classe do documento">
          ${options.map((name, i) => `<button type="button" class="rot-class" data-class="${esc(name)}" aria-pressed="${name === current}">
            <kbd>${KEYS[i] ?? ""}</kbd><span>${esc(name === UNMAPPED ? "sem rótulo" : name)}</span></button>`).join("")}
        </div>
        <p class="muted small">Atalhos: tecla da classe, <kbd>Enter</kbd> mantém a classe atual, <kbd>←</kbd>/<kbd>→</kbd> navegam.</p>
        <div class="rot-save">
          <button class="btn btn--primary" id="rot-flush" ${pending.size && !busy ? "" : "disabled"}>${busy ? "Salvando…" : `Salvar no git (${fmt(pending.size)})`}</button>
          ${pending.size ? `<button class="link" id="rot-discard">Descartar pendentes</button>` : ""}
          <p class="rot-msg" role="status">${esc(message)}</p>
        </div>`;
    side.querySelector("#rot-prev").addEventListener("click", () => { position--; message = ""; draw(); });
    side.querySelector("#rot-next").addEventListener("click", () => { position++; message = ""; draw(); });
    side.querySelectorAll("[data-class]").forEach((b) => b.addEventListener("click", () => decide(b.dataset.class)));
    side.querySelector("#rot-flush").addEventListener("click", flush);
    side.querySelector("#rot-discard")?.addEventListener("click", () => { pending.clear(); persistPending(); message = "Pendentes descartados."; draw(); });
    if (!readPassword() && pending.size) side.querySelector(".rot-save").prepend(passwordForm(() => { message = "Senha guardada nesta aba."; draw(); }));
  }

  function decide(name) {
    const doc = items[position];
    const original = saved.get(doc.id) ?? (queueKey === "auditoria" ? cls : doc.atual || UNMAPPED);
    if (name === original) pending.delete(doc.id);
    else pending.set(doc.id, changeFor(doc, name, "documento"));
    persistPending();
    message = "";
    if (position < items.length - 1) position++;
    if (pending.size >= AUTO_FLUSH && readPassword()) flush();
    else draw();
  }

  async function flush() {
    if (busy || !pending.size) return;
    busy = true;
    draw();
    const batch = [...pending.values()];
    const result = await send(batch, signal);
    if (signal.aborted) return;
    busy = false;
    if (result.ok) {
      for (const c of batch) { pending.delete(c.id); saved.set(c.id, c.class); }
      persistPending();
      if (result.applied.length) channel.post(result.applied);
      message = result.commit ? `${fmt(result.applied.length)} rótulos gravados no git (${result.commit.slice(0, 7)}).` : "Nada mudou.";
    } else message = result.error;
    draw();
  }

  const onKey = (e) => {
    if (e.target.closest?.("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
    const i = KEYS.indexOf(e.key.toLowerCase());
    if (i >= 0 && i < options.length) { e.preventDefault(); decide(options[i]); }
    else if (e.key === "Enter" && items.length) { e.preventDefault(); decide(currentOf(items[position])); }
    else if (e.key === "ArrowRight" && position < items.length - 1) { position++; message = ""; draw(); }
    else if (e.key === "ArrowLeft" && position > 0) { position--; message = ""; draw(); }
  };
  const onLeave = (e) => { if (pending.size) { e.preventDefault(); e.returnValue = ""; } };
  document.addEventListener("keydown", onKey);
  window.addEventListener("beforeunload", onLeave);

  Promise.all([loadQueue(cls, signal), loadSaved(signal)]).then(([queue, s]) => {
    saved = s;
    if (queueKey === "rejeitados") {
      const ids = new Set(readJson(`${REJECTED_KEY}:${cls}`, []));
      items = queue.candidatos.filter((d) => ids.has(d.id));
    } else items = queue[queueKey];
    position = Math.max(0, items.findIndex((d) => decided(d) === undefined));
    draw();
  }).catch((err) => { if (!signal.aborted) root.innerHTML = `<div class="notice">${esc(err.message)}</div>`; });

  return () => {
    abort.abort();
    channel.close();
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("beforeunload", onLeave);
  };
}

// ---------------------------------------------------------------- modo pasta
export function mountFolderMode(app, index, cls) {
  const abort = new AbortController();
  const { signal } = abort;
  const rejectedKey = `${REJECTED_KEY}:${cls}`;
  const rejectedEver = new Set(readJson(rejectedKey, []));
  let all = [], saved = new Map(), page = [], marked = new Set(), message = "", busy = false;
  let observer = null;
  const channel = openChannel((events) => { for (const e of events) saved.set(e.id, e.class); });
  let pageSize = PAGE_SIZES.includes(readJson(PAGE_SIZE_KEY, 120)) ? readJson(PAGE_SIZE_KEY, 120) : 120;

  app.innerHTML = shell(index, "pasta", cls, `<div class="toolbar rot-folder-bar">
      <select id="rot-class" aria-label="Classe">${classOptions(index, cls)}</select>
      <select id="rot-size" aria-label="Documentos por página">${PAGE_SIZES.map((n) => `<option value="${n}" ${n === pageSize ? "selected" : ""}>${n} por página</option>`).join("")}</select>
      <span class="muted small" id="rot-left"></span>
    </div>
    <div id="rot-folder"><p class="muted">Carregando fila…</p></div>`);
  app.querySelector("#rot-class").addEventListener("change", (e) => { location.hash = `#/pasta/${e.target.value}`; });
  app.querySelector("#rot-size").addEventListener("change", (e) => {
    if (marked.size) { message = "Confirme ou pule a página antes de mudar o tamanho."; e.target.value = String(pageSize); draw(); return; }
    pageSize = Number(e.target.value);
    writeJson(PAGE_SIZE_KEY, pageSize);
    message = "";
    nextPage();
  });
  const root = app.querySelector("#rot-folder");

  const open = (doc) => {
    const url = safeUrl(doc.url);
    document.getElementById("modal-title").textContent = `${doc.id} · ${doc.fornecedor || ""}`;
    document.getElementById("modal-body").innerHTML = url
      ? `<iframe class="rot-modal-pdf" src="${esc(url)}#view=FitH" title="PDF ${esc(doc.id)}"></iframe>`
      : `<div class="notice">Sem PDF.</div>`;
    document.getElementById("modal").hidden = false;
  };

  function nextPage() {
    const remaining = all.filter((d) => !saved.has(d.id) && !rejectedEver.has(d.id) && !pending.has(d.id));
    page = remaining.slice(0, pageSize);
    marked = new Set();
    app.querySelector("#rot-left").textContent = `${fmt(remaining.length)} candidatos restantes de ${fmt(all.length)}`;
    draw();
  }

  function draw() {
    observer?.disconnect();
    if (!page.length) {
      root.innerHTML = `<div class="notice">Nenhum candidato restante nesta classe. ${message ? esc(message) : ""}</div>`;
      return;
    }
    const keep = page.length - marked.size;
    root.innerHTML = `<div class="rot-actions">
        <button class="btn btn--primary" id="rot-confirm" ${keep && !busy ? "" : "disabled"}>${busy ? "Salvando…" : `Confirmar ${fmt(keep)} como ${esc(cls)}`}</button>
        <button class="btn" id="rot-skip" ${busy ? "disabled" : ""}>Pular página</button>
        <span class="muted small">${fmt(marked.size)} tirados · clique para tirar, clique duplo para abrir</span>
        <p class="rot-msg" role="status">${esc(message)}</p>
      </div>
      <ul class="rot-grid">${page.map((d) => `<li class="rot-card ${marked.has(d.id) ? "is-out" : ""}" data-id="${esc(d.id)}">
        <button type="button" class="rot-card-hit" aria-pressed="${marked.has(d.id)}" aria-label="${marked.has(d.id) ? "Devolver" : "Tirar"} ${esc(d.id)}">
          <span class="rot-thumb"><span class="muted small">carregando…</span></span>
          <span class="rot-card-text"><span class="mono">${esc(d.id)} · ${sim(d.sim)}</span>
            <span class="rot-card-sup">${esc(d.fornecedor || "—")}</span>
            ${d.atual && d.atual !== cls ? `<span class="chip chip--warn">atual: ${esc(d.atual)}</span>` : ""}</span>
        </button>
        <button type="button" class="link rot-open" aria-label="Abrir ${esc(d.id)}">abrir</button>
      </li>`).join("")}</ul>`;
    const pass = !readPassword() ? passwordForm(() => { message = "Senha guardada nesta aba."; draw(); }) : null;
    if (pass) root.querySelector(".rot-actions").append(pass);
    root.querySelector("#rot-confirm").addEventListener("click", confirm);
    root.querySelector("#rot-skip").addEventListener("click", () => { all = all.filter((d) => !page.includes(d)).concat(page); message = ""; nextPage(); });
    root.querySelectorAll(".rot-card").forEach((card) => {
      const doc = page.find((d) => d.id === card.dataset.id);
      card.querySelector(".rot-card-hit").addEventListener("click", () => {
        if (marked.has(doc.id)) marked.delete(doc.id); else marked.add(doc.id);
        card.classList.toggle("is-out", marked.has(doc.id));
        card.querySelector(".rot-card-hit").setAttribute("aria-pressed", String(marked.has(doc.id)));
        const confirmButton = root.querySelector("#rot-confirm");
        confirmButton.textContent = `Confirmar ${fmt(page.length - marked.size)} como ${cls}`;
        confirmButton.disabled = busy || page.length === marked.size;
        root.querySelector(".rot-actions .small").textContent = `${fmt(marked.size)} tirados · clique para tirar, clique duplo para abrir`;
      });
      card.querySelector(".rot-card-hit").addEventListener("dblclick", () => open(doc));
      card.querySelector(".rot-open").addEventListener("click", () => open(doc));
    });
    observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        observer.unobserve(entry.target);
        const doc = page.find((d) => d.id === entry.target.dataset.id);
        const slot = entry.target.querySelector(".rot-thumb");
        const url = safeUrl(doc?.url);
        if (!url) { slot.innerHTML = `<span class="muted small">sem PDF</span>`; continue; }
        thumbnail(url, 200).then((src) => { slot.innerHTML = `<img src="${src}" alt="Página 1 do documento ${esc(doc.id)}">`; })
          .catch(() => { slot.innerHTML = `<span class="muted small">sem miniatura</span>`; });
      }
    }, { rootMargin: "400px" });
    root.querySelectorAll(".rot-card").forEach((card) => observer.observe(card));
  }

  async function confirm() {
    if (busy) return;
    const keep = page.filter((d) => !marked.has(d.id));
    const out = page.filter((d) => marked.has(d.id));
    if (!readPassword()) { message = "Informe a senha de edição."; draw(); return; }
    busy = true;
    draw();
    const result = await send(keep.map((d) => changeFor(d, cls, "pasta")), signal);
    if (signal.aborted) return;
    busy = false;
    if (!result.ok) { message = result.error; draw(); return; }
    for (const d of keep) saved.set(d.id, cls);
    if (result.applied.length) channel.post(result.applied);
    for (const d of out) rejectedEver.add(d.id);
    writeJson(rejectedKey, [...rejectedEver]);
    message = `${fmt(result.applied.length)} documentos gravados como ${cls}${result.commit ? ` (${result.commit.slice(0, 7)})` : ""}.` +
      (out.length ? ` ${fmt(out.length)} tirados ficam na fila “Tirados da pasta” do modo documento.` : "");
    nextPage();
  }

  Promise.all([loadQueue(cls, signal), loadSaved(signal)]).then(([queue, s]) => {
    saved = s;
    all = queue.candidatos;
    nextPage();
  }).catch((err) => { if (!signal.aborted) root.innerHTML = `<div class="notice">${esc(err.message)}</div>`; });

  return () => { abort.abort(); observer?.disconnect(); channel.close(); };
}
