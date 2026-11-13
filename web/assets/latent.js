// Espaço latente em Three.js: desenha todos os pontos e identifica o ponto sob o cursor pela GPU.
// A seleção renderiza o índice de cada ponto em uma janela de 5x5 px, porque percorrer 1,3 milhão
// de pontos em JavaScript a cada movimento do mouse travaria a página.
import {
  BufferAttribute, BufferGeometry, Color, NearestFilter, NormalBlending, OrbitControls, PerspectiveCamera,
  Points, RGBAFormat, Scene, ShaderMaterial, UnsignedByteType, Vector3, WebGLRenderer, WebGLRenderTarget,
} from "./vendor/three.bundle.js";

const DATA = "data/latent/";
const PICK_SIZE = 5;
const MAX_CLASSES = 32;
const MORPH_MS = 900;
const FLY_MS = 650;
const HOME = new Vector3(100, 62, 140);
const SHORT_NAMES = { metric: "ArcFace", dino: "DINOv2", colpali: "ColPali", combined: "Multimodal" };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n).toLocaleString("pt-BR");
const mb = (n) => (n / 1048576).toFixed(1).replace(".", ",");
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Baixa com progresso e descomprime. Testa a assinatura gzip porque a CDN pode já ter descomprimido.
async function fetchBinary(path, onProgress, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`${response.status} ao baixar ${path}`);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(value.length);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

const POINT_VERTEX = /* glsl */ `
  attribute vec3 aTarget;
  attribute float aClass;
  uniform float uMix, uPixel, uScale, uPick, uUnmappedAlpha;
  uniform float uVisible[${MAX_CLASSES}];
  uniform vec3 uColor[${MAX_CLASSES}];
  varying vec4 vColor;
  void main() {
    int c = int(aClass + 0.5);
    float visible = uVisible[c];
    if (visible < 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
    vec4 mv = modelViewMatrix * vec4(mix(position, aTarget, uMix), 1.0);
    gl_Position = projectionMatrix * mv;
    float near = clamp(uScale / max(-mv.z, 1.0), 0.45, 5.0);
    bool labeled = c > 0;
    gl_PointSize = (labeled ? 2.6 : 1.3) * near * uPixel;
    if (uPick > 0.5) {
      gl_PointSize = max(gl_PointSize, labeled ? 5.0 : 3.0);
      // Rótulados ficam à frente dos não mapeados no teste de profundidade, como aparecem na tela.
      gl_Position.z = gl_Position.z * 0.5 + gl_Position.w * (labeled ? -0.5 : 0.5);
      float id = float(gl_VertexID + 1);
      vColor = vec4(mod(id, 256.0), mod(floor(id / 256.0), 256.0), floor(id / 65536.0), 255.0) / 255.0;
    } else {
      vColor = vec4(uColor[c], labeled ? 0.92 : uUnmappedAlpha);
    }
  }`;

const POINT_FRAGMENT = /* glsl */ `
  uniform float uPick;
  varying vec4 vColor;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = dot(p, p);
    if (d > 0.25) discard;
    gl_FragColor = uPick > 0.5 ? vColor : vec4(vColor.rgb, vColor.a * (1.0 - smoothstep(0.12, 0.25, d)));
  }`;

const MARK_VERTEX = /* glsl */ `
  uniform float uSize;
  void main() { gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = uSize; }`;
const MARK_FRAGMENT = /* glsl */ `
  uniform vec3 uTint;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = smoothstep(0.62, 0.70, d) * (1.0 - smoothstep(0.88, 1.0, d)) + 1.0 - smoothstep(0.22, 0.30, d);
    if (a < 0.01) discard;
    gl_FragColor = vec4(uTint, a);
  }`;

function makeMark(size, hex) {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new BufferAttribute(new Float32Array(3), 3));
  const material = new ShaderMaterial({
    vertexShader: MARK_VERTEX, fragmentShader: MARK_FRAGMENT, transparent: true, depthTest: false, depthWrite: false,
    uniforms: { uSize: { value: size }, uTint: { value: new Color(hex) } },
  });
  const mark = new Points(geometry, material);
  mark.frustumCulled = false;
  mark.renderOrder = 2;
  mark.visible = false;
  return mark;
}

const documentFromRow = (row) => ({ id: row[0], name: `${row[0]}.pdf`, fornecedor: row[1], subcota: row[2], ano: row[3], url: row[4] });
const safeUrl = (url) => (/^https:\/\//.test(url ?? "") ? url : null);
const PASSWORD_KEY = "ceap-editor-password";
const STRENGTH_KEY = "ceap-unmapped-strength";
const DEFAULT_STRENGTH = 33;
const clampStrength = (value) => (Number.isFinite(value) ? Math.min(100, Math.max(5, value)) : DEFAULT_STRENGTH);
const readStrength = () => {
  try { const saved = localStorage.getItem(STRENGTH_KEY); return saved === null ? DEFAULT_STRENGTH : clampStrength(Number(saved)); }
  catch { return DEFAULT_STRENGTH; }
};
// Quadrática: com milhões de pontos sobrepostos, a diferença visual se concentra nas opacidades baixas.
const strengthToAlpha = (value) => (value / 100) ** 2;
const readPassword = () => { try { return sessionStorage.getItem(PASSWORD_KEY); } catch { return null; } };
const writePassword = (value) => {
  try { if (value) sessionStorage.setItem(PASSWORD_KEY, value); else sessionStorage.removeItem(PASSWORD_KEY); } catch { /* sem armazenamento */ }
};
const apiClass = (c) => (c.id === 0 ? "_unmapped" : c.name);

function shell(index) {
  const labeled = index.classes.reduce((sum, c) => sum + (c.id ? c.count : 0), 0);
  const models = Object.values(index.models);
  return `<div class="lat" id="lat">
    <div class="lat-stage" id="lat-stage"></div>
    <header class="lat-head panel">
      <h1>Espaço latente</h1>
      <p class="muted lat-facts"><span class="mono">${fmt(index.n_points)}</span> documentos em 3D · <span class="mono" id="lat-labeled">${fmt(labeled)}</span> rotulados</p>
      <div class="segmented" role="radiogroup" aria-label="Espaço de representação">
        ${models.map((m) => `<button type="button" role="radio" data-model="${esc(m.id)}" aria-checked="false">${esc(SHORT_NAMES[m.id] ?? m.name)}</button>`).join("")}
      </div>
      <p class="lat-desc" id="lat-desc"></p>
    </header>
    <details class="lat-legend panel" id="lat-legend" open>
      <summary>Classes</summary>
      <div class="lat-legend-actions"><button type="button" class="link" id="lat-all">Mostrar todas</button></div>
      <ul id="lat-classes"></ul>
      <div class="lat-strength">
        <label for="lat-strength">Pontos não mapeados</label>
        <output id="lat-strength-value" for="lat-strength" class="mono"></output>
        <input type="range" id="lat-strength" min="5" max="100" step="1">
      </div>
    </details>
    <aside class="lat-doc panel" id="lat-doc" aria-live="polite" hidden></aside>
    <button type="button" class="btn lat-reset" id="lat-reset">Recentrar</button>
    <p class="lat-hint muted">Arraste para girar, role para aproximar e clique em um ponto para ver o documento.</p>
    <div class="lat-tip" id="lat-tip" role="status" hidden></div>
    <div class="lat-load" id="lat-load" role="status">
      <strong id="lat-load-title">Carregando pontos</strong>
      <div class="bar"><i id="lat-load-bar"></i></div>
      <span class="muted mono" id="lat-load-text"></span>
    </div>
  </div>`;
}

export function mountLatent(root, index, getDataset) {
  const abort = new AbortController();
  const { signal } = abort;
  root.innerHTML = shell(index);
  const $ = (id) => root.querySelector(`#${id}`);
  const stage = $("lat-stage");
  const tip = $("lat-tip");
  const docPanel = $("lat-doc");
  const n = index.n_points;
  let disposeScene = () => {};

  const fail = (message) => {
    $("lat-load").innerHTML = `<strong>Não foi possível desenhar os pontos</strong>
      <span class="muted">${esc(message)}</span><a href="#/documentos">Ver a lista de documentos</a>`;
    $("lat-load").hidden = false;
  };

  const docsByPoint = new Map();
  getDataset().then((ds) => { for (const d of ds.docs) docsByPoint.set(d.coords_idx, d); }).catch(() => {});

  // Vínculos reconstruídos ou vindos de um CSV: arquivos por faixa de pontos, baixados sob demanda.
  const shards = new Map();
  const loadShard = (shard) => {
    if (!shards.has(shard)) {
      const name = String(shard).padStart(5, "0");
      shards.set(shard, fetchBinary(`${DATA}meta/${name}.json.gz`, null, signal)
        .then((buffer) => JSON.parse(new TextDecoder().decode(buffer)))
        .catch(() => { shards.delete(shard); return null; }));
    }
    return shards.get(shard);
  };
  async function docFor(point) {
    if (docsByPoint.has(point)) return docsByPoint.get(point);
    if (!index.shard_size) return null;
    const rows = await loadShard(Math.floor(point / index.shard_size));
    const row = rows?.[point % index.shard_size];
    if (!row) return null;
    if (row.c) return { candidates: row.c.map(documentFromRow) };
    return { ...documentFromRow(row), reconstructed: true };
  }

  (async () => {
    const coordsTotal = index.models[index.default_model];
    let received = 0;
    const expected = 15.2 * 1048576;
    const progress = (bytes) => {
      received += bytes;
      $("lat-load-bar").style.width = `${Math.min(100, (received / expected) * 100).toFixed(1)}%`;
      $("lat-load-text").textContent = `${mb(received)} MB de cerca de ${mb(expected)} MB`;
    };
    const [classBuffer, coordBuffer] = await Promise.all([
      fetchBinary(`${DATA}classes.bin.gz`, progress, signal),
      fetchBinary(`${DATA}${coordsTotal.file}`, progress, signal),
    ]);
    if (signal.aborted) return;
    $("lat-load-title").textContent = "Preparando a cena";
    await new Promise((resolve) => setTimeout(resolve));
    if (signal.aborted) return;
    disposeScene = startScene(new Uint8Array(classBuffer), new Float32Array(coordBuffer));
    $("lat-load").hidden = true;
  })().catch((err) => { if (!signal.aborted) fail(err.message); });

  function startScene(classes, firstCoords) {
    if (classes.length !== n || firstCoords.length !== n * 3) throw new Error("Os arquivos de pontos não têm o tamanho esperado.");

    let renderer;
    try { renderer = new WebGLRenderer({ antialias: false, powerPreference: "high-performance" }); }
    catch { throw new Error("Este navegador não oferece WebGL2."); }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor(new Color("#0f0f0f"));
    stage.appendChild(renderer.domElement);
    const canvas = renderer.domElement;
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `Nuvem 3D com ${fmt(n)} documentos. A lista completa está na página Documentos.`);

    // Não mapeados primeiro: os rotulados são desenhados por cima deles.
    const order = new Uint32Array(n);
    let head = 0;
    for (let i = 0; i < n; i++) if (classes[i] === 0) order[head++] = i;
    for (let i = 0; i < n; i++) if (classes[i] !== 0) order[head++] = i;
    const reorder = (source) => {
      const out = new Float32Array(n * 3);
      for (let v = 0; v < n; v++) {
        const p = order[v] * 3;
        out[v * 3] = source[p]; out[v * 3 + 1] = source[p + 1]; out[v * 3 + 2] = source[p + 2];
      }
      return out;
    };
    const vertexOfPoint = new Uint32Array(n);
    for (let v = 0; v < n; v++) vertexOfPoint[order[v]] = v;
    const baseClasses = classes.slice();
    const classAttr = new Uint8Array(n);
    for (let v = 0; v < n; v++) classAttr[v] = classes[order[v]];

    const coords = new Map();
    const coordRequests = new Map();
    coords.set(index.default_model, reorder(firstCoords));

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(coords.get(index.default_model), 3));
    geometry.setAttribute("aTarget", new BufferAttribute(coords.get(index.default_model), 3));
    geometry.setAttribute("aClass", new BufferAttribute(classAttr, 1));

    const colors = Array.from({ length: MAX_CLASSES }, () => new Color("#c8ccd2"));
    for (const c of index.classes) if (c.id < MAX_CLASSES) colors[c.id].set(c.hex).convertLinearToSRGB();
    const visible = new Array(MAX_CLASSES).fill(1);
    const uniforms = {
      uMix: { value: 0 }, uPixel: { value: renderer.getPixelRatio() }, uScale: { value: 1 }, uPick: { value: 0 }, uUnmappedAlpha: { value: strengthToAlpha(readStrength()) },
      uVisible: { value: visible }, uColor: { value: colors },
    };
    const material = new ShaderMaterial({
      vertexShader: POINT_VERTEX, fragmentShader: POINT_FRAGMENT, uniforms,
      transparent: true, depthTest: false, depthWrite: false, blending: NormalBlending,
    });
    const cloud = new Points(geometry, material);
    cloud.frustumCulled = false;
    const scene = new Scene();
    scene.add(cloud);
    const hoverMark = makeMark(18 * renderer.getPixelRatio(), "#ffffff");
    const pinMark = makeMark(26 * renderer.getPixelRatio(), "#dafa9c");
    scene.add(hoverMark, pinMark);

    const camera = new PerspectiveCamera(50, 1, 0.5, 2000);
    camera.position.copy(HOME);
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4;
    controls.maxDistance = 600;

    const pickTarget = new WebGLRenderTarget(PICK_SIZE, PICK_SIZE, {
      format: RGBAFormat, type: UnsignedByteType, minFilter: NearestFilter, magFilter: NearestFilter, depthBuffer: true,
    });
    const pickPixels = new Uint8Array(PICK_SIZE * PICK_SIZE * 4);
    const pickBlack = new Color(0, 0, 0);

    let width = 1, height = 1, dirty = true, frame = 0, dragging = false;
    let pointer = null, pickDue = false, hovered = -1, pinned = -1, model = index.default_model, morph = null, fly = null, pinToken = 0;

    const resize = () => {
      width = stage.clientWidth || 1;
      height = stage.clientHeight || 1;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      dirty = true;
    };
    const observer = new ResizeObserver(resize);
    observer.observe(stage);
    resize();

    const pointPosition = (point) => {
      const vertex = vertexOfPoint[point];
      const a = geometry.attributes.position.array;
      const b = geometry.attributes.aTarget.array;
      const t = uniforms.uMix.value;
      const i = vertex * 3;
      return new Vector3(a[i] + (b[i] - a[i]) * t, a[i + 1] + (b[i + 1] - a[i + 1]) * t, a[i + 2] + (b[i + 2] - a[i + 2]) * t);
    };
    const placeMark = (mark, point) => {
      mark.visible = point >= 0;
      if (point < 0) return;
      mark.geometry.attributes.position.array.set(pointPosition(point).toArray());
      mark.geometry.attributes.position.needsUpdate = true;
    };

    function pick(x, y) {
      const half = (PICK_SIZE - 1) / 2;
      const size = renderer.getPixelRatio();
      camera.setViewOffset(width, height, x - half, y - half, PICK_SIZE, PICK_SIZE);
      camera.updateProjectionMatrix();
      const saved = { transparent: material.transparent, depthTest: material.depthTest, depthWrite: material.depthWrite };
      material.transparent = false; material.depthTest = true; material.depthWrite = true;
      uniforms.uPick.value = 1; uniforms.uPixel.value = 1;
      hoverMark.visible = false; const pinWasVisible = pinMark.visible; pinMark.visible = false;
      const clear = new Color(); renderer.getClearColor(clear);
      const alpha = renderer.getClearAlpha();
      renderer.setClearColor(pickBlack, 0);
      renderer.setRenderTarget(pickTarget);
      renderer.clear();
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(pickTarget, 0, 0, PICK_SIZE, PICK_SIZE, pickPixels);
      renderer.setRenderTarget(null);
      renderer.setClearColor(clear, alpha);
      Object.assign(material, saved);
      uniforms.uPick.value = 0; uniforms.uPixel.value = size;
      pinMark.visible = pinWasVisible;
      camera.clearViewOffset();
      camera.updateProjectionMatrix();
      let best = -1, bestDistance = Infinity;
      for (let py = 0; py < PICK_SIZE; py++) {
        for (let px = 0; px < PICK_SIZE; px++) {
          const i = (py * PICK_SIZE + px) * 4;
          if (pickPixels[i + 3] === 0) continue;
          const distance = (px - half) ** 2 + (py - half) ** 2;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = pickPixels[i] + pickPixels[i + 1] * 256 + pickPixels[i + 2] * 65536 - 1;
          }
        }
      }
      return best >= 0 && best < n ? order[best] : -1;
    }

    const classOf = (point) => index.classes.find((c) => c.id === classes[point]) ?? { name: "—", hex: "#858b94" };
    const counts = new Map(index.classes.map((c) => [c.id, c.count]));
    function setPointClass(point, id) {
      if (classes[point] === id) return;
      counts.set(classes[point], counts.get(classes[point]) - 1);
      counts.set(id, (counts.get(id) ?? 0) + 1);
      classes[point] = id;
      const vertex = vertexOfPoint[point];
      const attribute = geometry.attributes.aClass;
      attribute.array[vertex] = id;
      attribute.addUpdateRange(vertex, 1);
      attribute.needsUpdate = true;
      dirty = true;
    }
    const dot = (hex) => `<span class="dot" style="background:${esc(hex)}"></span>`;
    let hoverToken = 0;

    function showTip(point) {
      hoverToken++;
      if (point < 0) { tip.hidden = true; canvas.style.cursor = "grab"; return; }
      canvas.style.cursor = "pointer";
      const c = classOf(point);
      const token = hoverToken;
      const draw = (doc) => {
        if (token !== hoverToken) return;
        tip.innerHTML = `<div class="lat-tip-class">${dot(c.hex)}${esc(c.name)}</div>
          ${doc?.candidates ? `<div class="muted">Vínculo incerto · ${doc.candidates.length} candidatos</div>`
            : doc ? `<div class="mono">${esc(doc.name)}</div>${doc.fornecedor ? `<div class="muted">${esc(doc.fornecedor)}</div>` : ""}`
            : `<div class="muted mono">Ponto ${fmt(point)}</div>`}`;
        tip.hidden = false;
        positionTip();
      };
      draw(docsByPoint.get(point) ?? null);
      if (!docsByPoint.has(point) && index.shard_size) docFor(point).then((doc) => { if (doc) draw(doc); });
    }
    function positionTip() {
      if (!pointer || tip.hidden) return;
      const margin = 16;
      const x = pointer.x + margin + tip.offsetWidth > width ? pointer.x - tip.offsetWidth - margin : pointer.x + margin;
      const y = pointer.y + margin + tip.offsetHeight > height ? pointer.y - tip.offsetHeight - margin : pointer.y + margin;
      tip.style.transform = `translate(${Math.max(0, x)}px, ${Math.max(0, y)}px)`;
    }

    function pin(point, flyTo) {
      pinned = point;
      placeMark(pinMark, point);
      dirty = true;
      const token = ++pinToken;
      if (point < 0) { docPanel.hidden = true; root.querySelector("#lat").classList.remove("has-doc"); return; }
      const c = classOf(point);
      const position = pointPosition(point);
      const draw = (doc) => {
        if (token !== pinToken) return;
        const row = (label, value) => (value === undefined || value === null || value === "" ? "" : `<dt>${label}</dt><dd>${esc(value)}</dd>`);
        const candidates = doc?.candidates ?? null;
        const current = classOf(point);
        const url = safeUrl(doc?.url);
        const title = candidates ? "Vínculo incerto" : doc?.name ?? "Documento sem vínculo";
        docPanel.innerHTML = `<div class="lat-doc-head"><div class="lat-doc-class">${dot(current.hex)}${esc(current.name)}</div>
            <button type="button" class="btn ghost" id="lat-doc-close" aria-label="Fechar documento">✕</button></div>
          <h2 class="mono">${esc(title)}</h2>
          <dl>
            ${row("Fornecedor", doc?.fornecedor)}${row("Subcota", doc?.subcota)}${row("Ano", doc?.ano)}
            ${row("Status", classes[point] > 0 ? "rotulado" : "não rotulado")}
            ${row("Cluster", doc?.closest_cluster)}
            ${row("Similaridade", doc?.similarity != null ? Number(doc.similarity).toFixed(4).replace(".", ",") : null)}
            ${row("Ponto", fmt(point))}
            <dt>Posição</dt><dd class="mono">${position.toArray().map((v) => v.toFixed(2).replace(".", ",")).join("  ")}</dd>
          </dl>
          ${url ? `<a class="btn btn--primary" href="${esc(url)}" target="_blank" rel="noopener">Abrir PDF</a>` : ""}
          ${candidates ? `<p class="muted lat-doc-note">Alguns PDFs deste trecho falharam no processamento, então o documento é um destes. A classe só pode ser corrigida em vínculos exatos.</p>
            <ul class="lat-cands">${candidates.map((c) => `<li>${safeUrl(c.url) ? `<a class="mono" href="${esc(c.url)}" target="_blank" rel="noopener">${esc(c.name)}</a>` : `<span class="mono">${esc(c.name)}</span>`}
              ${c.fornecedor ? `<span class="muted">${esc(c.fornecedor)}</span>` : ""}</li>`).join("")}</ul>` : ""}
          ${doc?.reconstructed ? `<p class="muted lat-doc-note">Vínculo reconstruído pela ordem dos documentos.</p>` : ""}
          ${doc ? "" : `<p class="muted lat-doc-note">Este ponto não tem documento associado.</p>`}
          ${doc?.id ? editorHtml(point) : ""}`;
        docPanel.querySelector("#lat-doc-close").addEventListener("click", () => pin(-1));
        if (doc?.id) bindEditor(doc, point);
        docPanel.hidden = false;
        root.querySelector("#lat").classList.add("has-doc");
      };
      redrawPinned = draw;
      currentDoc = null;
      const show = (doc) => { currentDoc = doc; draw(doc); };
      show(docsByPoint.get(point) ?? null);
      if (!docsByPoint.has(point) && index.shard_size) docFor(point).then((doc) => { if (doc && token === pinToken) show(doc); });
      if (flyTo) flyToward(position);
    }

    let redrawPinned = () => {};
    let currentDoc = null;
    let editorMessage = "";
    function editorHtml(point) {
      const current = classes[point];
      const corrected = current !== baseClasses[point];
      const message = editorMessage;
      editorMessage = "";
      return `<form class="lat-edit" id="lat-edit">
        <label for="lat-edit-class">Corrigir classe</label>
        <select id="lat-edit-class">${index.classes.map((c) => `<option value="${c.id}" ${c.id === current ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select>
        ${readPassword() ? "" : `<label for="lat-edit-pass">Senha de edição</label><input type="password" id="lat-edit-pass" autocomplete="current-password" required>`}
        <button type="submit" class="btn" id="lat-edit-save">Salvar classe</button>
        ${corrected ? `<button type="button" class="link" id="lat-edit-undo">Desfazer correção</button>` : ""}
        <p class="lat-edit-msg" id="lat-edit-msg" role="status">${esc(message)}</p>
      </form>`;
    }
    function bindEditor(doc, point) {
      const form = docPanel.querySelector("#lat-edit");
      const message = (text) => { docPanel.querySelector("#lat-edit-msg").textContent = text; };
      const submit = async (id, done) => {
        const previous = classes[point];
        const target = index.classes.find((c) => c.id === id);
        const password = readPassword() ?? form.querySelector("#lat-edit-pass")?.value;
        if (!password) { message("Informe a senha de edição."); return; }
        form.querySelectorAll("button").forEach((b) => { b.disabled = true; });
        setPointClass(point, id);
        refreshLegend();
        let failure = null;
        try {
          const meta = { fornecedor: doc.fornecedor, subcota: doc.subcota, ano: Number(doc.ano) || undefined, url: safeUrl(doc.url) ?? undefined };
          const response = await fetch("api/labels", {
            method: "POST", signal, headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password, changes: [{ id: doc.id, point, class: apiClass(target), mode: "latente", meta }] }),
          });
          if (response.status === 401) writePassword(null);
          if (!response.ok) failure = (await response.json().catch(() => null))?.error ?? "Não foi possível salvar.";
          else writePassword(password);
        } catch (err) {
          if (signal.aborted) return;
          failure = "Sem conexão com o serviço de rótulos.";
        }
        if (failure) setPointClass(point, previous);
        refreshLegend();
        editorMessage = failure ?? done;
        if (pinned === point) { placeMark(pinMark, point); redrawPinned(currentDoc ?? doc); }
      };
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        const id = Number(form.querySelector("#lat-edit-class").value);
        if (id === classes[point]) { message("Esta já é a classe do ponto."); return; }
        submit(id, "Correção salva.");
      });
      form.querySelector("#lat-edit-undo")?.addEventListener("click", () => submit(baseClasses[point], "Correção desfeita."));
    }
    function flyToward(position) {
      const direction = camera.position.clone().sub(controls.target).normalize();
      const distance = Math.min(camera.position.distanceTo(controls.target), 70);
      flyTo(position.clone(), position.clone().addScaledVector(direction, distance));
    }
    function flyTo(target, cameraPosition) {
      const end = { target, camera: cameraPosition };
      if (reducedMotion()) {
        controls.target.copy(end.target);
        camera.position.copy(end.camera);
        dirty = true;
        return;
      }
      fly = { start: performance.now(), fromTarget: controls.target.clone(), fromCamera: camera.position.clone(), ...end };
    }

    const local = (event) => {
      const box = canvas.getBoundingClientRect();
      return { x: event.clientX - box.left, y: event.clientY - box.top };
    };
    let pressed = null;
    const onMove = (event) => {
      pointer = local(event);
      if (dragging || morph) return;
      pickDue = true;
      positionTip();
    };
    const onLeave = () => { pointer = null; pickDue = false; hovered = -1; placeMark(hoverMark, -1); showTip(-1); dirty = true; };
    const onDown = (event) => { pressed = local(event); dragging = true; showTip(-1); };
    const onUp = (event) => {
      dragging = false;
      if (!pressed) return;
      const p = local(event);
      const moved = Math.hypot(p.x - pressed.x, p.y - pressed.y);
      pressed = null;
      if (moved > 5 || morph) return;
      const point = pick(p.x, p.y);
      if (point >= 0) pin(point, true);
      else pin(-1);
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onDown);
    window.addEventListener("pointerup", onUp);
    const onKey = (event) => { if (event.key === "Escape" && pinned >= 0) pin(-1); };
    document.addEventListener("keydown", onKey);
    const onControlsChange = () => { dirty = true; };
    controls.addEventListener("change", onControlsChange);

    function loop(now) {
      frame = requestAnimationFrame(loop);
      if (fly) {
        const t = reducedMotion() ? 1 : Math.min(1, (now - fly.start) / FLY_MS);
        const k = ease(t);
        controls.target.lerpVectors(fly.fromTarget, fly.target, k);
        camera.position.lerpVectors(fly.fromCamera, fly.camera, k);
        dirty = true;
        if (t === 1) fly = null;
      }
      if (morph) {
        const t = reducedMotion() ? 1 : Math.min(1, (now - morph.start) / MORPH_MS);
        uniforms.uMix.value = ease(t);
        if (t === 1) {
          geometry.setAttribute("position", geometry.attributes.aTarget);
          uniforms.uMix.value = 0;
          morph = null;
        }
        placeMark(pinMark, pinned);
        dirty = true;
      }
      controls.update();
      if (pickDue && pointer && !dragging && !morph && !fly) {
        pickDue = false;
        const point = pick(pointer.x, pointer.y);
        if (point !== hovered) {
          hovered = point;
          placeMark(hoverMark, point);
          showTip(point);
          dirty = true;
        }
      }
      if (dirty) {
        dirty = false;
        uniforms.uScale.value = 170 / Math.max(camera.position.distanceTo(controls.target), 1);
        renderer.render(scene, camera);
      }
    }
    frame = requestAnimationFrame(loop);

    async function setModel(id) {
      if (id === model || morph) return;
      syncModel(id);
      if (!coords.has(id)) {
        $("lat-load-title").textContent = "Baixando o modelo";
        $("lat-load-bar").style.width = "0%";
        $("lat-load").hidden = false;
        let received = 0;
        const expected = 15.2 * 1048576;
        const request = coordRequests.get(id) ?? fetchBinary(`${DATA}${index.models[id].file}`, (bytes) => {
          received += bytes;
          $("lat-load-bar").style.width = `${Math.min(100, (received / expected) * 100).toFixed(1)}%`;
          $("lat-load-text").textContent = `${mb(received)} MB de cerca de ${mb(expected)} MB`;
        }, signal);
        coordRequests.set(id, request);
        let buffer;
        try { buffer = await request; }
        catch (err) {
          coordRequests.delete(id);
          if (!signal.aborted) { syncModel(model); $("lat-load").hidden = true; }
          return;
        }
        if (signal.aborted) return;
        coords.set(id, reorder(new Float32Array(buffer)));
        $("lat-load").hidden = true;
      }
      if (selectedModel !== id) return;
      model = id;
      geometry.setAttribute("aTarget", new BufferAttribute(coords.get(id), 3));
      hovered = -1;
      placeMark(hoverMark, -1);
      showTip(-1);
      morph = { start: performance.now() };
    }

    let selectedModel = model;
    function syncModel(id) {
      selectedModel = id;
      root.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.model === id)));
      $("lat-desc").textContent = index.models[id].description ?? "";
    }
    root.querySelectorAll("[data-model]").forEach((b) => b.addEventListener("click", () => setModel(b.dataset.model)));
    syncModel(model);

    // Legenda: a cor fica presa à classe; ligar, desligar ou isolar nunca recolore as demais.
    const list = $("lat-classes");
    function refreshLegend() {
      const rows = index.classes.filter((c) => counts.get(c.id) > 0)
        .sort((x, y) => (x.id === 0) - (y.id === 0) || counts.get(y.id) - counts.get(x.id));
      list.innerHTML = rows.map((c) => `<li><button type="button" class="lat-class" data-class="${c.id}" aria-pressed="${visible[c.id] === 1}">
        ${dot(c.hex)}<span class="lat-class-name">${esc(c.name)}</span><span class="muted mono">${fmt(counts.get(c.id))}</span></button>
        <button type="button" class="link lat-only" data-only="${c.id}" aria-label="Mostrar só ${esc(c.name)}">Só esta</button></li>`).join("");
      let labeled = 0;
      for (const [id, count] of counts) if (id !== 0) labeled += count;
      $("lat-labeled").textContent = fmt(labeled);
    }
    refreshLegend();
    const syncLegend = () => {
      list.querySelectorAll("[data-class]").forEach((b) => b.setAttribute("aria-pressed", String(visible[Number(b.dataset.class)] === 1)));
      if (hovered >= 0 && !visible[classes[hovered]]) { hovered = -1; placeMark(hoverMark, -1); showTip(-1); }
      if (pinned >= 0 && !visible[classes[pinned]]) pin(-1);
      pickDue = true;
      dirty = true;
    };
    list.addEventListener("click", (event) => {
      const toggle = event.target.closest("[data-class]");
      const only = event.target.closest("[data-only]");
      if (toggle) {
        const id = Number(toggle.dataset.class);
        visible[id] = visible[id] ? 0 : 1;
      } else if (only) {
        visible.fill(0);
        visible[Number(only.dataset.only)] = 1;
      } else return;
      syncLegend();
    });
    $("lat-all").addEventListener("click", () => { visible.fill(1); syncLegend(); });
    const strength = $("lat-strength");
    const showStrength = () => { $("lat-strength-value").textContent = `${strength.value}%`; };
    strength.value = String(readStrength());
    showStrength();
    strength.addEventListener("input", () => {
      uniforms.uUnmappedAlpha.value = strengthToAlpha(Number(strength.value));
      showStrength();
      dirty = true;
      try { localStorage.setItem(STRENGTH_KEY, strength.value); } catch { /* sem armazenamento */ }
    });
    $("lat-reset").addEventListener("click", () => flyTo(new Vector3(), HOME.clone()));
    if (matchMedia("(max-width: 760px)").matches) $("lat-legend").open = false;

    // Rótulos gravados pelo site (log no git): vale o evento mais recente de cada documento.
    fetch("api/labels", { signal, cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((data) => {
      if (!data) return;
      const idByApiName = new Map(index.classes.map((c) => [apiClass(c), c.id]));
      const latest = new Map();
      for (const event of data.corrections) latest.set(event.id, event);
      for (const event of latest.values()) {
        const id = idByApiName.get(event.class);
        if (id !== undefined && event.point >= 0 && event.point < n) setPointClass(event.point, id);
      }
      refreshLegend();
      stage.dataset.corrections = String(latest.size);
    }).catch(() => {});

    stage.dataset.points = String(geometry.attributes.position.count);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointerup", onUp);
      document.removeEventListener("keydown", onKey);
      controls.removeEventListener("change", onControlsChange);
      controls.dispose();
      geometry.dispose();
      material.dispose();
      hoverMark.geometry.dispose(); hoverMark.material.dispose();
      pinMark.geometry.dispose(); pinMark.material.dispose();
      pickTarget.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    };
  }

  return () => {
    abort.abort();
    disposeScene();
  };
}
