#!/home/lucas.ocunha/masters/.venv_ocr/bin/python3
"""
Isola os 669 Falsos Positivos Detectados pelo OCR para Validação Separada do Usuário.
- Move/remove os 669 documentos das pastas de lotes em validacao_nfse_38k/:
    • 459 DANFE NF-e (produtos / mod 55)
    • 199 Boletos Bancários
    • 11 Cupons Fiscais / NFC-e (mod 65 / SAT)
- Cria a pasta estruturada no Desktop:
    Desktop/validacao_falsos_positivos_ocr/
        ├── 1_danfe_nfe/          (459 imagens)
        ├── 2_boletos/            (199 imagens)
        ├── 3_cupons_nfce/        (11 imagens)
        ├── manifest.json
        └── README.txt
- Pré-renderiza thumbnails para o Nautilus/GNOME exibir instantaneamente.
"""

import os
import sys
import json
import gzip
import shutil
import hashlib
import urllib.parse
from pathlib import Path
from collections import Counter
import pymupdf

BASE_DIR = Path("/home/lucas.ocunha/masters")
DATA_DIR = BASE_DIR / "data"
VALIDATION_DIR = DATA_DIR / "validacao_batches" / "validacao_nfse_38k"
RESULTS_FILE = VALIDATION_DIR / "rapidocr_results.json.gz"
RENDERED_DIR = DATA_DIR / "rendered_pages"
DEST_DIR = DATA_DIR / "validacao_batches" / "validacao_falsos_positivos_ocr"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop")
DESKTOP_LINK = DESKTOP_DIR / "validacao_falsos_positivos_ocr"
THUMB_DIR = Path(os.path.expanduser("~/.cache/thumbnails/large"))
THUMB_DIR.mkdir(parents=True, exist_ok=True)


def setup_thumbnail(image_path: Path):
    """Gera thumbnail no cache do GNOME Nautilus para visualização instantânea."""
    try:
        uri = "file://" + urllib.parse.quote(str(image_path.resolve()))
        h = hashlib.md5(uri.encode("utf-8")).hexdigest()
        thumb_p = THUMB_DIR / f"{h}.png"
        if thumb_p.exists():
            return True
        doc = pymupdf.open(str(image_path))
        if len(doc) == 0:
            doc.close()
            return False
        pix = doc[0].get_pixmap(dpi=72)
        pix.save(str(thumb_p))
        doc.close()
        return True
    except Exception:
        return False


def main():
    print("=" * 72)
    print(" ISOLAMENTO DOS FALSOS POSITIVOS DETECTADOS PELO OCR")
    print("=" * 72)

    # 1. Carregar resultados do RapidOCR
    print(f"\n1. Carregando {RESULTS_FILE}...")
    with gzip.open(RESULTS_FILE, "rt", encoding="utf-8") as f:
        results = json.load(f)
    print(f"   Total de documentos no histórico: {len(results):,}")

    # Filtrar falsos positivos
    target_classes = {
        "danfe_nfe": "1_danfe_nfe",
        "boleto": "2_boletos",
        "cupom_nfce": "3_cupons_nfce"
    }

    false_positives = [
        v for v in results.values()
        if v.get("classe") in target_classes
    ]
    print(f"\n2. Falsos positivos identificados para isolamento: {len(false_positives):,}")
    counts = Counter(v["classe"] for v in false_positives)
    for c, cnt in counts.items():
        print(f"   • {c:15s}: {cnt:4,}")

    # 2. Criar diretórios de destino
    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for sub in target_classes.values():
        (DEST_DIR / sub).mkdir(parents=True, exist_ok=True)

    # 3. Mover/remover dos lotes de validacao_nfse_38k e hardlink para destino
    print("\n3. Removendo dos lotes de NFS-e e organizando na pasta de validação...")
    manifest_records = []
    removed_from_lotes = 0
    linked_to_dest = 0

    for doc in false_positives:
        doc_id = str(doc["doc_id"])
        classe = doc["classe"]
        subfolder = target_classes[classe]
        tier = doc.get("tier", "")
        lote = doc.get("lote", "")

        dest_png = DEST_DIR / subfolder / f"{doc_id}.png"
        src_rendered = RENDERED_DIR / f"{doc_id}.png"

        # Procurar no lote original em validacao_nfse_38k
        lote_png = VALIDATION_DIR / tier / lote / f"{doc_id}.png"
        if lote_png.exists():
            # Remove do lote original para despoluir
            lote_png.unlink()
            removed_from_lotes += 1

        # Criar hardlink na pasta de destino (ou copiar do rendered_pages se necessário)
        if src_rendered.exists():
            if not dest_png.exists():
                try:
                    os.link(src_rendered, dest_png)
                except Exception:
                    shutil.copy2(src_rendered, dest_png)
            linked_to_dest += 1
            setup_thumbnail(dest_png)

        manifest_records.append({
            "doc_id": doc_id,
            "doc_name": doc.get("doc_name", f"{doc_id}.pdf"),
            "classe_detectada": classe,
            "subpasta": subfolder,
            "fornecedor": doc.get("fornecedor", ""),
            "tier_origem": tier,
            "lote_origem": lote,
            "similarity_nfse": doc.get("similarity", 0.0),
            "trecho_ocr": (doc.get("texto", "") or "")[:200]
        })

    print(f"   ✓ {removed_from_lotes:,} arquivos removidos das pastas em validacao_nfse_38k/")
    print(f"   ✓ {linked_to_dest:,} imagens organizadas com thumbnails em {DEST_DIR.name}/")

    # 4. Gravar manifesto e README
    print("\n4. Gravando manifest.json e instruções...")
    with open(DEST_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2, ensure_ascii=False)

    readme_content = f"""VALIDAÇÃO DE FALSOS POSITIVOS DETECTADOS PELO OCR
==================================================
Total de Documentos Isolados: {len(manifest_records):,}

Pastas:
  1_danfe_nfe/   : {counts.get('danfe_nfe', 0):,} documentos identificados como DANFE (mercadorias/produtos mod 55)
  2_boletos/     : {counts.get('boleto', 0):,} documentos identificados como Boletos Bancários
  3_cupons_nfce/ : {counts.get('cupom_nfce', 0):,} documentos identificados como Cupons Fiscais / NFC-e

Como Validar:
- Navegue pelas imagens e confirme se todas são de fato falsos positivos (DANFE, Boletos ou Cupons).
- Caso encontre alguma NFS-e legítima por engano, você pode simplesmente movê-la de volta ou deletá-la se preferir.
- Quando terminar a checagem, basta avisar o agente para registrar o resultado final no histórico.
"""
    with open(DEST_DIR / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 5. Criar atalho no Desktop
    print("\n5. Criando atalho no Desktop...")
    if DESKTOP_LINK.exists() or DESKTOP_LINK.is_symlink():
        DESKTOP_LINK.unlink()
    DESKTOP_LINK.symlink_to(DEST_DIR)
    print(f"   ✓ Atalho criado com sucesso em: {DESKTOP_LINK}")

    # 6. Atualizar resumo de validacao_nfse_38k
    remaining_pngs = len(list(VALIDATION_DIR.glob("*/*/*.png")))
    summary_nfse_file = VALIDATION_DIR / "manifest_summary.json"
    if summary_nfse_file.exists():
        with open(summary_nfse_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["total_candidatos_restantes_lotes"] = remaining_pngs
        summary["falsos_positivos_removidos_ocr"] = len(false_positives)
        with open(summary_nfse_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(" ISOLAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Total isolado em 'validacao_falsos_positivos_ocr': {len(false_positives):,}")
    print(f"Total restante limpo em 'validacao_nfse_38k': {remaining_pngs:,}")
    print("=" * 72)


if __name__ == "__main__":
    main()
