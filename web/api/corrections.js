// Correções de classe do espaço latente. Cada correção é um evento gravado no Vercel Blob, com os dados
// no próprio caminho: listar basta para ler tudo, e o histórico de eventos nunca é sobrescrito.
import { createHash, timingSafeEqual } from "node:crypto";

export const UNMAPPED = "_unmapped";
export const CLASSES = [
  UNMAPPED, "agua", "boleto", "correios", "cupom_fiscal", "emails", "energia", "fatura", "hotelaria", "iptu",
  "nfce", "nfe", "nfse", "passagem_aerea", "passagem_rodoviaria", "pedagio", "recibo", "sem_parar", "telefonia", "uber",
];
const MAX_POINT = 1339677;
const JSON_HEADERS = { "Content-Type": "application/json", "Cache-Control": "no-store" };
const reply = (status, body) => new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });

const digest = (value) => createHash("sha256").update(String(value)).digest();
const samePassword = (given, expected) => timingSafeEqual(digest(given), digest(expected));
const prefixOf = (env) => env.CORRECTIONS_PREFIX || "corrections/";

export const toPathname = (prefix, { id, point, class: name, ts }) => `${prefix}${id}/${ts}_${point}_${name}.json`;
export function fromPathname(pathname) {
  const match = pathname.match(/([0-9]+)\/([0-9]+)_([0-9]+)_([a-z_]+)\.json$/);
  return match && { id: match[1], ts: Number(match[2]), point: Number(match[3]), class: match[4] };
}

const blobStore = {
  async list(prefix) {
    const { list } = await import("@vercel/blob");
    const pathnames = [];
    let cursor;
    do {
      const page = await list({ prefix, cursor });
      pathnames.push(...page.blobs.map((b) => b.pathname));
      cursor = page.hasMore ? page.cursor : undefined;
    } while (cursor);
    return pathnames;
  },
  async put(pathname, body) {
    const { put } = await import("@vercel/blob");
    await put(pathname, body, { access: "private", addRandomSuffix: false, contentType: "application/json" });
  },
};

export async function listCorrections(env, store) {
  try {
    const events = (await store.list(prefixOf(env))).map(fromPathname).filter(Boolean);
    return reply(200, { corrections: events.sort((a, b) => a.ts - b.ts) });
  } catch (err) {
    console.error(err);
    return reply(502, { error: "Não foi possível ler as correções." });
  }
}

export async function saveCorrection(request, env, store, now = Date.now) {
  if (!env.EDITOR_PASSWORD) return reply(503, { error: "A edição está desligada neste ambiente." });
  let body;
  try { body = await request.json(); } catch { return reply(400, { error: "Corpo inválido." }); }
  if (typeof body?.password !== "string" || !samePassword(body.password, env.EDITOR_PASSWORD)) {
    return reply(401, { error: "Senha incorreta." });
  }
  const { id, point, class: name } = body;
  if (typeof id !== "string" || !/^[0-9]{1,12}$/.test(id)) return reply(400, { error: "Documento inválido." });
  if (!Number.isInteger(point) || point < 0 || point > MAX_POINT) return reply(400, { error: "Ponto inválido." });
  if (!CLASSES.includes(name)) return reply(400, { error: "Classe inválida." });
  const event = { id, point, class: name, ts: now() };
  try {
    await store.put(toPathname(prefixOf(env), event), JSON.stringify(event));
  } catch (err) {
    console.error(err);
    return reply(502, { error: "Não foi possível gravar a correção." });
  }
  return reply(201, event);
}

export const GET = () => listCorrections(process.env, blobStore);
export const POST = (request) => saveCorrection(request, process.env, blobStore);
