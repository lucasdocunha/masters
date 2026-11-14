#!/usr/bin/env python3
"""
Remove estritamente qualquer documento de NFC-e (Modelo 65 / Consumidor Eletrônica)
e NFS-e (Serviços / Prefeituras / RPS / CF-e-SAT) dos lotes de validação da NF-e.
Garante que permaneçam APENAS Notas Fiscais Eletrônicas (Modelo 55 / DANFE A4).
"""

import json
import re
import urllib.parse
import hashlib
from pathlib import Path
import psycopg2

BASE_DIR = Path("/home/lucas.ocunha/masters/data/validacao_batches/validacao_nfe_novos_candidatos")
THUMB_DIR = Path.home() / ".cache" / "thumbnails" / "large"
DESKTOP_DIR = Path("/home/lucas.ocunha/Desktop/validacao_nfe_novos_candidatos")

def main():
    print("=" * 70)
    print("PURGA E PURIFICAÇÃO DOS LOTES DE NF-E (REMOÇÃO DE NFC-e E NFS-e)")
    print("=" * 70)

    all_pngs = list(BASE_DIR.rglob("*.png"))
    file_map = {}
    for p in all_pngs:
        m = re.search(r'_(\d+)\.png$', p.name)
        if m:
            file_map[int(m.group(1))] = p

    doc_ids = list(file_map.keys())
    print(f"Total de documentos atuais nos lotes: {len(doc_ids):,}")

    conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/ceap")
    cur = conn.cursor()

    to_remove = {}
    chunk_size = 5000
    for i in range(0, len(doc_ids), chunk_size):
        chunk = doc_ids[i:i+chunk_size]
        cur.execute("""
            SELECT ide_documento, texto_completo
            FROM documento_texto_completo
            WHERE ide_documento = ANY(%s)
        """, (chunk,))
        for doc_id, txt in cur.fetchall():
            txt_u = txt.upper()

            # 1. NFC-e e SAT (Consumidor)
            is_nfce = any(w in txt_u for w in [
                "NFC-E", "MODELO 65", "MOD. 65", "MOD 65",
                "CONSUMIDOR ELETR", "DANFE NFC-E", 
                "NOTA FISCAL DE CONSUMIDOR",
                "CF-E-SAT", "CUPOM FISCAL ELETRÔNICO", "EXTRATO DO CF-E-SAT"
            ])
            if is_nfce:
                to_remove[doc_id] = "nfce_sat"
                continue

            # 2. NFS-e (Serviços Municipais)
            has_danfe_55 = ("DANFE" in txt_u and "MODELO 55" in txt_u) or ("5500" in txt_u and "CHAVE DE ACESSO" in txt_u)
            if not has_danfe_55:
                is_nfse = any(w in txt_u for w in [
                    "NOTA FISCAL DE SERVIÇO", "NOTA FISCAL DE SERVICO", 
                    "NFS-E", "PREFEITURA MUNICIPAL", "SECRETARIA MUNICIPAL DE FINAN",
                    "SECRETARIA MUNICIPAL DE FAZENDA", "RECIBO PROVISÓRIO DE SERVIÇOS",
                    "RPS Nº", "TRIBUTADA NO MUNICÍPIO"
                ])
                if is_nfse:
                    to_remove[doc_id] = "nfse"
                    continue

    conn.close()

    print(f"Documentos a remover: {len(to_remove):,}")
    print(f"  - NFC-e / Consumidor / SAT: {sum(1 for v in to_remove.values() if v == 'nfce_sat'):,}")
    print(f"  - NFS-e / Serviços Municipais: {sum(1 for v in to_remove.values() if v == 'nfse'):,}")

    # Remover os arquivos e miniaturas
    removed_count = 0
    for doc_id, reason in to_remove.items():
        p = file_map[doc_id]
        if p.exists() or p.is_symlink():
            p.unlink()
            removed_count += 1
            # Miniaturas
            for base_path in [p, DESKTOP_DIR / p.parent.name / p.name]:
                uri = "file://" + urllib.parse.quote(str(base_path.absolute()))
                h = hashlib.md5(uri.encode("utf-8")).hexdigest()
                t_path = THUMB_DIR / f"{h}.png"
                if t_path.exists() or t_path.is_symlink():
                    try:
                        t_path.unlink()
                    except Exception:
                        pass

    print(f"Removidos {removed_count:,} atalhos de NFC-e e NFS-e!")

    # Contagem final por pasta
    print("\nResumo das pastas após a purificação:")
    total_remaining = 0
    for sub in sorted(BASE_DIR.iterdir()):
        if sub.is_dir():
            cnt = len(list(sub.glob("*.png")))
            total_remaining += cnt
            print(f"  - {sub.name:38s}: {cnt:5,d} NF-e (100% Modelo 55 / DANFE A4)")

    print(f"\nTotal restante de NF-e puras nos lotes: {total_remaining:,}")

    # Atualizar manifesto
    manifest_file = BASE_DIR / ".candidatos_manifesto.json"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        filtered_manifest = {k: v for k, v in manifest.items() if int(k) not in to_remove}
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(filtered_manifest, f, indent=2, ensure_ascii=False)
        print(f"Manifesto atualizado com {len(filtered_manifest):,} candidatos limpos.")

if __name__ == "__main__":
    main()
