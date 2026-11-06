// Rodar com: node --test tests/web/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { CLASSES, fromPathname, listCorrections, saveCorrection, toPathname, UNMAPPED } from "../../web/api/corrections.js";

const env = { EDITOR_PASSWORD: "senha-de-teste", CORRECTIONS_PREFIX: "t/" };
const memory = () => {
  const paths = [];
  return { paths, list: async (prefix) => paths.filter((p) => p.startsWith(prefix)), put: async (p) => { paths.push(p); } };
};
const post = (body) => new Request("http://x/api/corrections", { method: "POST", body: JSON.stringify(body) });
const valid = { password: "senha-de-teste", id: "5541335", point: 7551, class: "nfe" };

test("grava e lista a correção", async () => {
  const store = memory();
  const saved = await saveCorrection(post(valid), env, store, () => 1000);
  assert.equal(saved.status, 201);
  assert.deepEqual(store.paths, ["t/5541335/1000_7551_nfe.json"]);
  const listed = await (await listCorrections(env, store)).json();
  assert.deepEqual(listed.corrections, [{ id: "5541335", ts: 1000, point: 7551, class: "nfe" }]);
});

test("a resposta nunca devolve a senha", async () => {
  const text = await (await saveCorrection(post(valid), env, memory())).text();
  assert.ok(!text.includes("senha-de-teste"));
});

test("senha errada ou ausente retorna 401 e não grava", async () => {
  for (const password of ["errada", undefined, 123]) {
    const store = memory();
    assert.equal((await saveCorrection(post({ ...valid, password }), env, store)).status, 401);
    assert.equal(store.paths.length, 0);
  }
});

test("sem senha configurada a edição fica desligada", async () => {
  const store = memory();
  assert.equal((await saveCorrection(post(valid), {}, store)).status, 503);
  assert.equal(store.paths.length, 0);
});

test("dados inválidos retornam 400", async () => {
  for (const bad of [{ class: "x" }, { class: "../nfe" }, { id: "../1" }, { id: 1 }, { point: -1 }, { point: 1339678 }, { point: 1.5 }]) {
    const store = memory();
    assert.equal((await saveCorrection(post({ ...valid, ...bad }), env, store)).status, 400, JSON.stringify(bad));
    assert.equal(store.paths.length, 0);
  }
});

test("JSON malformado retorna 400", async () => {
  const request = new Request("http://x", { method: "POST", body: "{" });
  assert.equal((await saveCorrection(request, env, memory())).status, 400);
});

test("caminho ida e volta, com classe sem rótulo", () => {
  const event = { id: "1", point: 2, class: UNMAPPED, ts: 3 };
  assert.deepEqual(fromPathname(toPathname("corrections/", event)), event);
  assert.equal(fromPathname("corrections/lixo.txt"), null);
});

test("as classes da API são as do índice do site", () => {
  const index = JSON.parse(readFileSync(new URL("../../web/data/latent/index.json", import.meta.url)));
  const names = index.classes.map((c) => (c.id === 0 ? UNMAPPED : c.name));
  assert.deepEqual([...names].sort(), [...CLASSES].sort());
});
