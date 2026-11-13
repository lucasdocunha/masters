// Rodar com: node --test tests/web/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  applyChanges, appendLog, CLASSES, commitMessage, Conflict, CSV_HEADER, githubStore, LOG_HEADER, listLabels, newRow,
  parseLog, partOf, saveLabels, UNMAPPED,
} from "../../web/api/labels.js";

const HEADER = CSV_HEADER;
const PART = "data/rotulos/parte_0000.csv";
const CSV = [
  HEADER,
  `5,https://www.camara.leg.br/cota-parlamentar/documentos/publ/1/2014/5.pdf,"{""fornecedor"":""A, B""}",2014,nfe,rotulado`,
  `20,,"{}",2015,uber,rotulado`,
  "",
].join("\n");
const env = { EDITOR_PASSWORD: "senha-de-teste" };
const url = "https://www.camara.leg.br/cota-parlamentar/documentos/publ/9/2020/7.pdf";

// GitHub em memória: arquivos por commit, com uma branch que pode "andar" para simular conflito.
function fakeGithub(files = { [PART]: CSV }, { conflicts = 0 } = {}) {
  const commits = { c0: { files: { ...files } } };
  const blobs = {};
  const state = { head: "c0", n: 0, conflicts };
  return {
    state, commits,
    head: async () => state.head,
    treeOf: async (c) => `t-${c}`,
    read: async (path, ref) => commits[ref].files[path] ?? null,
    blob: async (content) => { const sha = `b${++state.n}`; blobs[sha] = content; return sha; },
    tree: async (base, list) => ({ base, list }),
    commit: async (message, tree, parent) => {
      const sha = `c${++state.n}`;
      const files = { ...commits[parent].files };
      for (const { path, sha: b } of tree.list) { if (b === null) delete files[path]; else files[path] = blobs[b]; }
      commits[sha] = { files, message, parent };
      return sha;
    },
    advance: async (sha) => {
      if (state.conflicts > 0) {
        state.conflicts--;
        const other = `x${state.n}`;
        commits[other] = { files: { ...commits[state.head].files } };
        state.head = other;
        throw new Conflict("mudou");
      }
      state.head = sha;
    },
  };
}
const post = (changes, password = "senha-de-teste") =>
  new Request("http://x/api/labels", { method: "POST", body: JSON.stringify({ password, changes }) });
const fileAtHead = (gh, path = PART) => gh.commits[gh.state.head].files[path];

test("troca a classe só no fim da linha e mantém os metadados com vírgula", () => {
  const { text, applied } = applyChanges(CSV, [{ id: "5", class: "boleto" }]);
  assert.equal(text.split("\n")[1], `5,https://www.camara.leg.br/cota-parlamentar/documentos/publ/1/2014/5.pdf,"{""fornecedor"":""A, B""}",2014,boleto,rotulado`);
  assert.deepEqual(applied, [{ id: "5", class: "boleto", from: "nfe" }]);
  assert.ok(text.endsWith("\n"));
});

test("documento novo entra em ordem numérica; _unmapped remove a linha", () => {
  const { text, applied } = applyChanges(CSV, [
    { id: "7", class: "agua", meta: { fornecedor: 'X "Y"', subcota: "S", ano: 2020, url } },
    { id: "20", class: UNMAPPED },
  ]);
  const lines = text.trim().split("\n");
  assert.deepEqual(lines.map((l) => l.split(",")[0]), ["doc_id", "5", "7"]);
  assert.equal(lines[2], newRow("7", "agua", { fornecedor: 'X "Y"', subcota: "S", ano: 2020, url }));
  assert.match(lines[2], /^7,https:.*7\.pdf,"\{""fornecedor"":""X \\""Y\\""""/);
  assert.deepEqual(applied.map((c) => c.from), ["", "uber"]);
});

test("mudança para a mesma classe não altera nada", () => {
  assert.deepEqual(applyChanges(CSV, [{ id: "5", class: "nfe" }, { id: "99", class: UNMAPPED }]), { text: CSV, applied: [] });
});

test("URL fora da Câmara não vai para o CSV", () => {
  assert.equal(newRow("1", "agua", { url: "https://evil.example/1.pdf" }).split(",")[1], "");
});

test("log: cria cabeçalho, anexa e lê de volta", () => {
  const log = appendLog("", [{ id: "5", point: 3, class: "boleto", from: "nfe", mode: "pasta" }], 1000);
  assert.equal(log, `${LOG_HEADER}\n1000,5,3,nfe,boleto,pasta\n`);
  const more = appendLog(log, [{ id: "8", class: UNMAPPED, from: "", mode: "documento" }], 2000);
  assert.deepEqual(parseLog(more), [
    { id: "5", point: 3, class: "boleto", ts: 1000, mode: "pasta" },
    { id: "8", point: -1, class: UNMAPPED, ts: 2000, mode: "documento" },
  ]);
});

test("POST grava CSV e log num único commit", async () => {
  const gh = fakeGithub();
  const response = await saveLabels(post([
    { id: "5", class: "boleto", point: 10, mode: "pasta" },
    { id: "7", class: "boleto", point: 11, mode: "pasta", meta: { fornecedor: "F", ano: 2020, url } },
  ]), env, gh, () => 1234);
  assert.equal(response.status, 201);
  const body = await response.json();
  assert.equal(body.applied.length, 2);
  assert.equal(gh.commits[body.commit].parent, "c0");
  assert.match(gh.commits[body.commit].message, /^rotulagem\(site\): 2 docs \[pasta\]/);
  assert.match(fileAtHead(gh), /\n5,.*,boleto,rotulado\n7,.*,boleto,rotulado\n20,/);
  assert.equal(fileAtHead(gh, "data/labeling/correcoes_site.csv"), `${LOG_HEADER}\n1234,5,10,nfe,boleto,pasta\n1234,7,11,_unmapped,boleto,pasta\n`);
  assert.ok(!JSON.stringify(body).includes("senha-de-teste"));
});

test("sem mudanças efetivas não cria commit", async () => {
  const gh = fakeGithub();
  const response = await saveLabels(post([{ id: "5", class: "nfe", mode: "documento" }]), env, gh);
  assert.equal(response.status, 200);
  assert.equal(gh.state.head, "c0");
});

test("refaz a partir da nova ponta quando a branch anda; desiste depois de 3", async () => {
  const gh = fakeGithub(undefined, { conflicts: 2 });
  const ok = await saveLabels(post([{ id: "5", class: "agua", mode: "documento" }]), env, gh);
  assert.equal(ok.status, 201);
  assert.match(gh.commits[gh.state.head].parent, /^x/);
  const stuck = fakeGithub(undefined, { conflicts: 3 });
  assert.equal((await saveLabels(post([{ id: "5", class: "agua", mode: "documento" }]), env, stuck)).status, 409);
});

test("senha errada, edição desligada ou git não configurado não gravam", async () => {
  const gh = fakeGithub();
  const change = [{ id: "5", class: "agua", mode: "documento" }];
  for (const password of ["errada", null, 123]) assert.equal((await saveLabels(post(change, password), env, gh)).status, 401);
  assert.equal((await saveLabels(post(change), {}, gh)).status, 503);
  assert.equal((await saveLabels(post(change), env, null)).status, 503);
  assert.equal(gh.state.head, "c0");
});

test("dados inválidos retornam 400", async () => {
  const ok = { id: "5", class: "agua", mode: "documento" };
  const bad = [
    [], [{ ...ok, class: "x" }], [{ ...ok, id: "../1" }], [{ ...ok, id: 5 }], [{ ...ok, point: 1.5 }],
    [{ ...ok, point: 1339678 }], [{ ...ok, mode: "x" }], [ok, ok], [{ ...ok, meta: "x" }],
    Array.from({ length: 501 }, (_, i) => ({ ...ok, id: String(i) })),
  ];
  for (const changes of bad) {
    const gh = fakeGithub();
    assert.equal((await saveLabels(post(changes), env, gh)).status, 400, JSON.stringify(changes).slice(0, 80));
    assert.equal(gh.state.head, "c0");
  }
  assert.equal((await saveLabels(new Request("http://x", { method: "POST", body: "{" }), env, fakeGithub())).status, 400);
});

test("GET devolve o log como correções; sem git, lista vazia", async () => {
  const gh = fakeGithub({ "data/labeling/correcoes_site.csv": `${LOG_HEADER}\n5,1,2,nfe,agua,latente\n` });
  assert.deepEqual((await (await listLabels(env, gh)).json()).corrections, [{ id: "1", point: 2, class: "agua", ts: 5, mode: "latente" }]);
  assert.deepEqual((await (await listLabels(env, null)).json()).corrections, []);
});

test("cliente GitHub: conflito na ref vira Conflict; arquivo ausente vira null", async () => {
  const calls = [];
  const fakeFetch = async (href, init) => {
    calls.push([init.method ?? "GET", href]);
    if (href.includes("/contents/")) return new Response("", { status: 404 });
    return new Response("{}", { status: 422 });
  };
  const store = githubStore({ GITHUB_REPO: "o/r", GITHUB_TOKEN: "t", GITHUB_BRANCH: "teste" }, fakeFetch);
  assert.equal(await store.read("a/b.csv", "abc"), null);
  await assert.rejects(store.advance("sha"), Conflict);
  assert.deepEqual(calls[1], ["PATCH", "https://api.github.com/repos/o/r/git/refs/heads/teste"]);
});

test("commitMessage resume por classe", () => {
  const msg = commitMessage([{ id: "1", class: "agua", from: "", mode: "pasta" }, { id: "2", class: UNMAPPED, from: "nfe", mode: "documento" }]);
  assert.match(msg, /^rotulagem\(site\): 2 docs \[pasta, documento\]\n\n1 agua, 1 sem rótulo\n\n1: _unmapped -> agua\n2: nfe -> _unmapped$/);
});

test("as classes da API são as do índice do site", () => {
  const index = JSON.parse(readFileSync(new URL("../../web/data/latent/index.json", import.meta.url)));
  const names = index.classes.map((c) => (c.id === 0 ? UNMAPPED : c.name));
  assert.ok(names.every((name) => CLASSES.includes(name)));
});

test("parte de cada documento: doc_id // 25000 com 4 dígitos", () => {
  assert.equal(partOf("data/rotulos", "5451737"), "data/rotulos/parte_0218.csv");
  assert.equal(partOf("data/rotulos", "24999"), "data/rotulos/parte_0000.csv");
  assert.equal(partOf("data/rotulos", "25000"), "data/rotulos/parte_0001.csv");
});

test("um commit altera só as partes afetadas, cria parte nova e apaga parte vazia", async () => {
  const other = "data/rotulos/parte_0002.csv";
  const gh = fakeGithub({ [PART]: CSV, [other]: `${HEADER}\n50000,,"{}",2020,nfe,rotulado\n`, "data/rotulos/parte_0009.csv": `${HEADER}\n` });
  const response = await saveLabels(post([
    { id: "5", class: "agua", mode: "pasta" },
    { id: "50000", class: UNMAPPED, mode: "pasta" },
    { id: "30000", class: "nfe", mode: "pasta", meta: { url } },
  ]), env, gh, () => 1);
  assert.equal(response.status, 201);
  const files = gh.commits[gh.state.head].files;
  assert.match(files[PART], /\n5,.*,agua,rotulado\n20,/);
  assert.equal(other in files, false);
  assert.equal(files["data/rotulos/parte_0001.csv"], `${HEADER}\n${newRow("30000", "nfe", { url })}\n`);
  assert.equal(files["data/rotulos/parte_0009.csv"], `${HEADER}\n`);
  assert.deepEqual((await response.json()).applied.map((c) => c.id), ["5", "50000", "30000"]);
});

test("o cabeçalho do JS é o das partes versionadas", () => {
  const part = readFileSync(new URL("../../data/rotulos/parte_0218.csv", import.meta.url), "utf8");
  assert.equal(part.slice(0, part.indexOf("\n")), CSV_HEADER);
});
