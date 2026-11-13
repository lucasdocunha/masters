// Rótulos do site gravados direto no git. Cada envio vira um commit que altera, juntos:
// - data/rotulos/parte_NNNN.csv: as partes do CSV de rotulados que contêm os documentos (fonte da verdade);
// - data/labeling/correcoes_site.csv: uma linha por mudança (auditoria e sobreposição de classes no site).
// O CSV é dividido em partes de 25.000 doc_id porque a API do GitHub recusa arquivos acima de ~40 MB.
// O formato é um contrato com src/dados/labeling/labels_csv.py: parte = doc_id // 25000, uma linha por
// documento, ordenada por doc_id, e `classe_mapeada,status` sempre como os dois últimos campos, sem aspas.
import { createHash, timingSafeEqual } from "node:crypto";

export const UNMAPPED = "_unmapped";
export const CLASSES = [
  UNMAPPED, "agua", "boleto", "correios", "cupom_fiscal", "emails", "energia", "fatura", "hotelaria", "iptu",
  "nfce", "nfe", "nfse", "passagem_aerea", "passagem_rodoviaria", "pedagio", "recibo", "sem_parar", "telefonia", "uber",
];
export const MODES = ["documento", "pasta", "latente"];
export const MAX_CHANGES = 500;
export const LOG_HEADER = "ts,doc_id,ponto,classe_anterior,classe_nova,modo";
export const CSV_HEADER = "doc_id,link_documento,metadados_json,ano,classe_mapeada,status";
export const SPAN = 25000;
export const PDF_URL = /^https:\/\/www\.camara\.leg\.br\/cota-parlamentar\/documentos\/publ\/\d+\/\d{4}\/\d+\.pdf$/;
const STATUS = "rotulado";
const MAX_POINT = 1339677;
const RETRIES = 3;
const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });

const digest = (value) => createHash("sha256").update(String(value)).digest();
const samePassword = (given, expected) => timingSafeEqual(digest(given), digest(expected));
const paths = (env) => ({
  dir: (env.LABELS_DIR || "data/rotulos").replace(/\/+$/, ""),
  log: env.LABELS_LOG_PATH || "data/labeling/correcoes_site.csv",
});
export const partOf = (dir, id) => `${dir}/parte_${String(Math.floor(Number(id) / SPAN)).padStart(4, "0")}.csv`;

// ---------- CSV ----------
const idOf = (line) => line.slice(0, line.indexOf(","));
const classOf = (line) => {
  const end = line.lastIndexOf(",");
  return line.slice(line.lastIndexOf(",", end - 1) + 1, end);
};
const withClass = (line, name) => `${line.slice(0, line.lastIndexOf(",", line.lastIndexOf(",") - 1))},${name},${STATUS}`;
const quote = (value) => `"${String(value).replace(/"/g, '""')}"`;
const clean = (value) => String(value ?? "").replace(/\s+/g, " ").trim().slice(0, 300);
const byId = (a, b) => {
  const x = idOf(a), y = idOf(b);
  return x.length - y.length || (x < y ? -1 : x > y ? 1 : 0);
};

export function newRow(id, name, meta = {}) {
  const info = JSON.stringify({ fornecedor: clean(meta.fornecedor), subcota: clean(meta.subcota) });
  const link = PDF_URL.test(meta.url ?? "") ? meta.url : "";
  const ano = Number.isInteger(meta.ano) ? meta.ano : "";
  return `${id},${link},${quote(info)},${ano},${name},${STATUS}`;
}

// Aplica as mudanças ao texto do CSV. Devolve o novo texto e o que de fato mudou (com a classe anterior).
export function applyChanges(text, changes) {
  const trailing = text.endsWith("\n");
  const lines = (trailing ? text.slice(0, -1) : text).split("\n");
  const header = lines.shift();
  const where = new Map(lines.map((line, i) => [idOf(line), i]));
  const applied = [];
  let added = false;
  for (const change of changes) {
    const i = where.get(change.id);
    const before = i === undefined || lines[i] === null ? "" : classOf(lines[i]);
    const after = change.class === UNMAPPED ? "" : change.class;
    if (before === after) continue;
    if (!after) lines[i] = null;
    else if (before) lines[i] = withClass(lines[i], after);
    else {
      where.set(change.id, lines.length);
      lines.push(newRow(change.id, after, change.meta));
      added = true;
    }
    applied.push({ ...change, from: before });
  }
  const rows = lines.filter((line) => line !== null);
  if (added) rows.sort(byId);
  return { text: [header, ...rows].join("\n") + (trailing ? "\n" : ""), applied };
}

export function appendLog(text, applied, ts) {
  const base = text && text.trim() ? (text.endsWith("\n") ? text : `${text}\n`) : `${LOG_HEADER}\n`;
  const lines = applied.map((c) => [ts, c.id, c.point ?? -1, c.from || UNMAPPED, c.class, c.mode].join(","));
  return base + lines.map((line) => `${line}\n`).join("");
}

export function parseLog(text) {
  const events = [];
  for (const line of (text ?? "").split("\n").slice(1)) {
    const [ts, id, point, , name, mode] = line.split(",");
    if (!id || !CLASSES.includes(name)) continue;
    events.push({ id, point: Number(point), class: name, ts: Number(ts), mode });
  }
  return events.sort((a, b) => a.ts - b.ts);
}

// ---------- validação ----------
export function validateChanges(body) {
  const list = body?.changes;
  if (!Array.isArray(list) || list.length === 0) return "Nenhuma mudança enviada.";
  if (list.length > MAX_CHANGES) return `No máximo ${MAX_CHANGES} mudanças por envio.`;
  const seen = new Set();
  for (const c of list) {
    if (typeof c?.id !== "string" || !/^[0-9]{1,12}$/.test(c.id)) return "Documento inválido.";
    if (seen.has(c.id)) return `Documento repetido: ${c.id}.`;
    seen.add(c.id);
    if (!CLASSES.includes(c.class)) return "Classe inválida.";
    if (c.point !== undefined && (!Number.isInteger(c.point) || c.point < -1 || c.point > MAX_POINT)) return "Ponto inválido.";
    if (!MODES.includes(c.mode)) return "Modo inválido.";
    if (c.meta !== undefined && (typeof c.meta !== "object" || c.meta === null)) return "Metadados inválidos.";
  }
  return null;
}

const tidy = (c) => ({
  id: c.id, class: c.class, point: c.point ?? -1, mode: c.mode,
  meta: c.meta && { fornecedor: c.meta.fornecedor, subcota: c.meta.subcota, ano: c.meta.ano, url: c.meta.url },
});

export function commitMessage(applied) {
  const modes = [...new Set(applied.map((c) => c.mode))].join(", ");
  const counts = new Map();
  for (const c of applied) counts.set(c.class, (counts.get(c.class) ?? 0) + 1);
  const parts = [...counts].sort((a, b) => b[1] - a[1]).map(([name, n]) => `${n} ${name === UNMAPPED ? "sem rótulo" : name}`);
  const head = `rotulagem(site): ${applied.length} doc${applied.length > 1 ? "s" : ""} [${modes}]`;
  return `${head}\n\n${parts.join(", ")}\n\n${applied.map((c) => `${c.id}: ${c.from || UNMAPPED} -> ${c.class}`).join("\n")}`;
}

// ---------- GitHub ----------
export class Conflict extends Error {}

export function githubStore(env, fetchImpl = fetch) {
  const repo = env.GITHUB_REPO;
  const branch = env.GITHUB_BRANCH || "main";
  const api = async (path, init = {}, accept = "application/vnd.github+json") => {
    const response = await fetchImpl(`https://api.github.com/repos/${repo}${path}`, {
      ...init,
      headers: {
        Accept: accept, Authorization: `Bearer ${env.GITHUB_TOKEN}`, "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ceap-rotulagem", ...(init.body ? { "Content-Type": "application/json" } : {}),
      },
    });
    if (response.status === 404 && accept.endsWith("raw")) return null;
    if (response.status === 422 && path.startsWith("/git/refs/")) throw new Conflict("A branch mudou.");
    if (!response.ok) throw new Error(`GitHub ${response.status} em ${path}: ${(await response.text()).slice(0, 200)}`);
    return accept.endsWith("raw") ? response.text() : response.json();
  };
  const post = (path, body, method = "POST") => api(path, { method, body: JSON.stringify(body) });
  return {
    head: async () => (await api(`/git/ref/heads/${branch}`)).object.sha,
    treeOf: async (commit) => (await api(`/git/commits/${commit}`)).tree.sha,
    read: (path, ref) => api(`/contents/${path}?ref=${ref}`, {}, "application/vnd.github.raw"),
    blob: async (content) => (await post("/git/blobs", { content, encoding: "utf-8" })).sha,
    tree: async (base, files) => (await post("/git/trees", {
      base_tree: base, tree: files.map(({ path, sha }) => ({ path, sha: sha ?? null, mode: "100644", type: "blob" })),
    })).sha,
    commit: async (message, tree, parent) => (await post("/git/commits", { message, tree, parents: [parent] })).sha,
    advance: (sha) => post(`/git/refs/heads/${branch}`, { sha, force: false }, "PATCH"),
  };
}

// ---------- handlers ----------
export async function saveLabels(request, env, store, now = Date.now) {
  if (!env.EDITOR_PASSWORD) return reply(503, { error: "A edição está desligada neste ambiente." });
  if (!store) return reply(503, { error: "Gravação no git não configurada (GITHUB_TOKEN/GITHUB_REPO)." });
  let body;
  try { body = await request.json(); } catch { return reply(400, { error: "Corpo inválido." }); }
  if (typeof body?.password !== "string" || !samePassword(body.password, env.EDITOR_PASSWORD)) {
    return reply(401, { error: "Senha incorreta." });
  }
  const problem = validateChanges(body);
  if (problem) return reply(400, { error: problem });
  const changes = body.changes.map(tidy);
  const { dir, log } = paths(env);
  const byPart = new Map();
  for (const change of changes) {
    const part = partOf(dir, change.id);
    byPart.set(part, [...(byPart.get(part) ?? []), change]);
  }

  for (let attempt = 1; ; attempt++) {
    try {
      const parent = await store.head();
      const parts = [...byPart.keys()];
      const [logText, ...texts] = await Promise.all([store.read(log, parent), ...parts.map((p) => store.read(p, parent))]);
      const edited = [];
      const appliedAll = [];
      parts.forEach((part, i) => {
        const result = applyChanges(texts[i] ?? `${CSV_HEADER}\n`, byPart.get(part));
        if (!result.applied.length) return;
        appliedAll.push(...result.applied);
        // Parte que ficou só com o cabeçalho é apagada (sha nulo remove o arquivo da árvore).
        edited.push({ path: part, text: result.text.trim() === CSV_HEADER ? null : result.text });
      });
      if (!appliedAll.length) return reply(200, { commit: null, applied: [] });
      const order = new Map(changes.map((c, i) => [c.id, i]));
      appliedAll.sort((a, b) => order.get(a.id) - order.get(b.id));
      const ts = now();
      const [base, logSha, ...shas] = await Promise.all([
        store.treeOf(parent), store.blob(appendLog(logText, appliedAll, ts)),
        ...edited.map((e) => (e.text === null ? null : store.blob(e.text))),
      ]);
      const files = [...edited.map((e, i) => ({ path: e.path, sha: shas[i] })), { path: log, sha: logSha }];
      const tree = await store.tree(base, files);
      const commit = await store.commit(commitMessage(appliedAll), tree, parent);
      await store.advance(commit);
      const applied = appliedAll.map(({ id, point, class: name, from, mode }) => ({ id, point, class: name, from: from || UNMAPPED, mode, ts }));
      return reply(201, { commit, applied });
    } catch (err) {
      if (err instanceof Conflict && attempt < RETRIES) continue;
      console.error(err);
      return reply(err instanceof Conflict ? 409 : 502, { error: "Não foi possível gravar no git. Tente de novo." });
    }
  }
}

export async function listLabels(env, store) {
  if (!store) return reply(200, { corrections: [] });
  try {
    const head = await store.head();
    return reply(200, { corrections: parseLog(await store.read(paths(env).log, head)) });
  } catch (err) {
    console.error(err);
    return reply(502, { error: "Não foi possível ler as correções." });
  }
}

const storeFor = (env) => (env.GITHUB_TOKEN && env.GITHUB_REPO ? githubStore(env) : null);
export const GET = () => listLabels(process.env, storeFor(process.env));
export const POST = (request) => saveLabels(request, process.env, storeFor(process.env));
