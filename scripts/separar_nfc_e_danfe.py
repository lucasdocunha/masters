#!/usr/bin/env python3
"""
Analisa os documentos da pasta 'validacao_nfs' e separa automaticamente:
- NFC-e (Modelo 65 / Cupons fiscais / Combustíveis)
- DANFE (Modelo 55 / NF-e mercantil / Produtos / Editoras)
- Outros / Recibos / Faturas de locação
- NFS-e (Notas Fiscais de Serviços Municipais legítimas)

Pode mover os arquivos não-NFS para subpastas para manter a pasta principal 100% limpa.
"""

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pymupdf

FOLDER = Path("/home/lucas.ocunha/Desktop/validacao_nfs")
MANIFEST_FILE = FOLDER / ".candidatos_iniciais.json"


def analyze_doc(item):
    pdf_path = Path(item["pdf_path"])
    text = ""
    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) > 0:
            text = doc[0].get_text()
        doc.close()
    except Exception:
        pass

    t = text.upper()
    forn = item.get("fornecedor", "").upper()
    subc = item.get("subcota", "").upper()

    # Extrai chave de 44 dígitos
    clean_digits = re.sub(r"\D", "", t)
    model = None
    for k in re.findall(r"\d{44}", clean_digits):
        m = k[20:22]
        if m in ("55", "65"):
            model = m
            break

    # 1. NFC-e
    if model == "65":
        return "NFC_E", "Chave de acesso modelo 65 (NFC-e de consumidor)"
    if "NFC-E" in t or "DANFE NFC" in t or "CONSUMIDOR ELETRÔNICA" in t or "CONSUMIDOR ELETRONICA" in t or "EXTRATO SAT" in t:
        return "NFC_E", "Texto contém NFC-e / Cupom de Consumidor Eletrônica"
    if "COMBUSTÍVEIS E LUBRIFICANTES" in subc:
        if "DANFE" in t and "DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA" in t and ("ENTRADA" in t or "SAÍDA" in t or "SAIDA" in t):
            return "DANFE_NFE", "NF-e mercantil de combustível (DANFE Modelo 55)"
        return "NFC_E", "Posto de combustíveis (Cupom / NFC-e)"

    # 2. DANFE (NF-e modelo 55)
    if model == "55":
        return "DANFE_NFE", "Chave de acesso modelo 55 (NF-e mercantil de produtos)"
    if ("DANFE" in t and ("DOCUMENTO AUXILIAR" in t or "ENTRADA" in t or "SAÍDA" in t or "SAIDA" in t)) or ("DOCUMENTO AUXILIAR DA NOTA FISCAL ELETRÔNICA" in t):
        return "DANFE_NFE", "Layout oficial DANFE (NF-e modelo 55)"
    if "ASSINATURA DE PUBLICAÇÕES" in subc and ("EDITORA" in forn or "S/A" in forn or "S.A." in forn or "JORNAL" in forn):
        return "DANFE_NFE", "Editora / Publicações (NF-e mercantil modelo 55)"

    # 3. Faturas / Recibos de Locação
    if ("FATURA" in t or "RECIBO" in t) and not ("NFS-E" in t or "NOTA FISCAL" in t or "PREFEITURA" in t or "MUNICÍPIO" in t or "MUNICIPIO" in t):
        if "LOCAÇÃO" in subc or "LOCADORA" in forn or "TRANSPORTES" in forn:
            return "RECIBO_FATURA", "Fatura / Recibo de locação de veículos/bens móveis (sem ISS)"
        return "RECIBO_FATURA", "Recibo ou fatura avulsa (sem ISS municipal)"

    # 4. NFS-e Municipal
    if "NFS-E" in t or "NOTA FISCAL DE SERVIÇ" in t or "NOTA FISCAL DE SERVIC" in t or "PREFEITURA" in t or "IMPOSTO SOBRE SERVIÇ" in t or "ISS" in t:
        return "NFSE", "NFS-e confirmada por texto/prefeitura/ISS"

    if "DIVULGAÇÃO DA ATIVIDADE PARLAMENTAR" in subc or "HOSPEDAGEM" in subc or "CONSULTORIAS" in subc or "SEGURANÇA" in subc:
        return "NFSE", "Serviço prestado (subcota típica de NFS-e)"

    return "OUTROS", "Layout atípico / Comprovante avulso"


def main():
    parser = argparse.ArgumentParser(description="Identifica e separa NFC-e e DANFE na pasta validacao_nfs")
    parser.add_argument("--mover", action="store_true", help="Move os arquivos não-NFS para subpastas dentro de validacao_nfs")
    args = parser.parse_args()

    if not MANIFEST_FILE.exists():
        print(f"Erro: {MANIFEST_FILE} não encontrado.")
        sys.exit(1)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    current_files = {p.name: p for p in FOLDER.glob("*.pdf")}
    remaining = [m for m in manifest if m["filename"] in current_files]

    print(f"Analisando {len(remaining)} arquivos presentes em {FOLDER}...")

    def process(item):
        cat, reason = analyze_doc(item)
        res = dict(item)
        res["category"] = cat
        res["reason"] = reason
        return res

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process, remaining))

    nfse = [r for r in results if r["category"] == "NFSE"]
    nfce = [r for r in results if r["category"] == "NFC_E"]
    danfe = [r for r in results if r["category"] == "DANFE_NFE"]
    recibos = [r for r in results if r["category"] == "RECIBO_FATURA"]
    outros = [r for r in results if r["category"] == "OUTROS"]

    print("\n" + "="*50)
    print("RESUMO DA CLASSIFICAÇÃO:")
    print(f"  [OK] NFS-e legítimas (Serviços):  {len(nfse):4d} ({len(nfse)/len(results):.1%})")
    print(f"  [X]  NFC-e (Cupons/Combustíveis):  {len(nfce):4d} ({len(nfce)/len(results):.1%})")
    print(f"  [X]  DANFE (NF-e Mod 55 Mercantil): {len(danfe):4d} ({len(danfe)/len(results):.1%})")
    print(f"  [X]  Recibos / Faturas de locação: {len(recibos):4d} ({len(recibos)/len(results):.1%})")
    print(f"  [?]  Outros / Layouts atípicos:    {len(outros):4d} ({len(outros)/len(results):.1%})")
    print("="*50)

    if args.mover:
        print("\nMovendo documentos não-NFS para subpastas organizadas...")
        dirs = {
            "NFC_E": FOLDER / "_detectados_nfc",
            "DANFE_NFE": FOLDER / "_detectados_danfe",
            "RECIBO_FATURA": FOLDER / "_detectados_faturas_recibos",
            "OUTROS": FOLDER / "_detectados_outros",
        }
        for d in dirs.values():
            d.mkdir(exist_ok=True)

        moved = 0
        for r in results:
            cat = r["category"]
            if cat != "NFSE":
                src = FOLDER / r["filename"]
                dst = dirs[cat] / r["filename"]
                if src.exists() or src.is_symlink():
                    src.rename(dst)
                    moved += 1

        print(f"[SUCESSO] {moved} arquivos movidos para subpastas!")
        print(f"Restaram exatamente {len(nfse)} arquivos NFS-e limpos na pasta principal {FOLDER.name}!")


if __name__ == "__main__":
    main()
