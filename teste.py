# pip install -U transformers accelerate pymupdf pillow pandas
import io
from pathlib import Path
import urllib.request
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM
import pandas as pd

MODEL = "maritaca-ai/S5-nano"
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

prompt = """
Você é um especialista em extração de dados de documentos fiscais brasileiros (NF-e / DANFE / NFC-e). Analise a imagem do documento fornecido e extraia exclusivamente os campos solicitados, retornando APENAS um objeto JSON válido, sem markdown (```json) ou texto explicativo.

Regras estritas de extração:
1. Se um campo não estiver visível, estiver ilegível ou não existir no documento, retorne `null` (nunca invente ou estime valores).
2. Mantenha apenas dígitos (sem pontos, barras ou traços) nos campos numéricos de identificação.
3. Atenção à diferença entre Emitente/Remetente (quem emitiu a nota, geralmente no topo à esquerda) e Destinatário/Remetente (o cliente/comprador, no bloco "DESTINATÁRIO / REMETENTE").

Schema JSON esperado e regras por campo:
{
  "numero_nota": "String contendo apenas os dígitos do número da NF-e (ex: '000123456'). Ignore a série.",
  "chave_de_acesso": "String contendo exatamente os 44 dígitos numéricos da chave de acesso, sem espaços.",
  "data_emissao": "String no formato ISO YYYY-MM-DD (ex: '2026-11-15').",
  "valor_total_nota": "Número (float) usando ponto como separador decimal (ex: 1250.50). Corresponde ao campo 'VALOR TOTAL DA NOTA'.",
  "cnpj_emitente": "String com os 14 dígitos numéricos do CNPJ do emissor/fornecedor (cabeçalho da nota).",
  "cnpj_destinatario": "String com os 14 dígitos numéricos do CNPJ do destinatário/cliente (bloco Destinatário/Remetente). Se for CPF, retorne os 11 dígitos."
}
"""


def baixar_imagens(url, dpi=150, paginas=None):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        dados = resp.read()

    if dados[:4] == b"%PDF" or url.lower().endswith(".pdf"):
        doc = fitz.open(stream=dados, filetype="pdf")
        idx = paginas or range(len(doc))
        imgs = []
        for i in idx:
            pix = doc[i].get_pixmap(dpi=dpi)
            imgs.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        doc.close()
        return imgs

    return [Image.open(io.BytesIO(dados)).convert("RGB")]


def perguntar(imagens, pergunta, pensar=False, max_new_tokens=1024):
    content = [{"type": "image", "image": img} for img in imagens]
    content.append({"type": "text", "text": pergunta})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
        enable_thinking=pensar,          # False = modo Instruct
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    novos = out[0][inputs["input_ids"].shape[1]:]   # só os tokens gerados
    return processor.decode(novos, skip_special_tokens=True)


df = pd.read_csv("data/ceap_documentos_rotulados.csv")
saida_dir = Path("saida_inferencias")
saida_dir.mkdir(exist_ok=True)

for idx, row in df.iterrows():
    doc_id = row.get("doc_id", idx)
    imgs = baixar_imagens(row["link_documento"])
    resposta = perguntar(imgs, prompt)

    (saida_dir / f"{doc_id}.json").write_text(resposta, encoding="utf-8")
    df.loc[idx, "resposta"] = resposta

df.to_csv("resultados.csv", index=False)

