// Rodar com: node --test tests/web/
import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_BYTES, proxyPdf } from "../../web/api/pdf.js";

const ok = "https://www.camara.leg.br/cota-parlamentar/documentos/publ/1001/2014/5457507.pdf";
const get = (url) => new Request(`http://x/api/pdf?url=${encodeURIComponent(url)}`);
const pdf = (body = "%PDF-1.4 x", init = {}) => async () => new Response(body, { status: 200, ...init });

test("repassa o PDF da Câmara com cache", async () => {
  const response = await proxyPdf(get(ok), pdf());
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("content-type"), "application/pdf");
  assert.equal(await response.text(), "%PDF-1.4 x");
});

test("recusa qualquer outro endereço sem buscar", async () => {
  let fetched = false;
  const spy = async () => { fetched = true; return new Response("%PDF-"); };
  for (const url of [
    "https://evil.example/cota-parlamentar/documentos/publ/1/2014/1.pdf", "http://www.camara.leg.br/cota-parlamentar/documentos/publ/1/2014/1.pdf",
    `${ok}?x=1`, "https://www.camara.leg.br/cota-parlamentar/documentos/publ/../../x.pdf", "",
  ]) assert.equal((await proxyPdf(get(url), spy)).status, 400, url);
  assert.equal(fetched, false);
});

test("redirecionamento, erro, não-PDF e arquivo grande não passam", async () => {
  assert.equal((await proxyPdf(get(ok), async () => new Response(null, { status: 302, headers: { location: "https://x" } }))).status, 502);
  assert.equal((await proxyPdf(get(ok), async () => new Response("", { status: 404 }))).status, 404);
  assert.equal((await proxyPdf(get(ok), pdf("<html>"))).status, 502);
  assert.equal((await proxyPdf(get(ok), pdf("%PDF-", { headers: { "content-length": String(MAX_BYTES + 1) } }))).status, 413);
  assert.equal((await proxyPdf(get(ok), pdf(`%PDF-${"x".repeat(MAX_BYTES)}`))).status, 413);
  assert.equal((await proxyPdf(get(ok), async () => { throw new Error("rede"); })).status, 502);
});
