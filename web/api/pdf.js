// Repassa PDFs da Câmara para o pdf.js desenhar miniaturas: o site da Câmara não envia CORS.
// Só aceita URLs de comprovantes da cota parlamentar e não segue redirecionamentos.
import { PDF_URL } from "./labels.js";

export const MAX_BYTES = 4_400_000; // limite de resposta das funções do Vercel (4,5 MB)

const fail = (status, error) => new Response(JSON.stringify({ error }), {
  status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
});

export async function proxyPdf(request, fetchImpl = fetch) {
  const target = new URL(request.url).searchParams.get("url") ?? "";
  if (!PDF_URL.test(target)) return fail(400, "URL de PDF inválida.");
  let upstream;
  try {
    upstream = await fetchImpl(target, { redirect: "manual", headers: { "User-Agent": "ceap-rotulagem" } });
  } catch {
    return fail(502, "A Câmara não respondeu.");
  }
  if (upstream.status !== 200) return fail(upstream.status === 404 ? 404 : 502, `A Câmara respondeu ${upstream.status}.`);
  const declared = Number(upstream.headers.get("content-length") ?? 0);
  if (declared > MAX_BYTES) return fail(413, "PDF grande demais para a miniatura.");
  const bytes = new Uint8Array(await upstream.arrayBuffer());
  if (bytes.length > MAX_BYTES) return fail(413, "PDF grande demais para a miniatura.");
  if (String.fromCharCode(...bytes.subarray(0, 5)) !== "%PDF-") return fail(502, "A resposta não é um PDF.");
  return new Response(bytes, {
    status: 200,
    headers: { "Content-Type": "application/pdf", "Cache-Control": "public, max-age=86400, s-maxage=604800, immutable" },
  });
}

export const GET = (request) => proxyPdf(request);
