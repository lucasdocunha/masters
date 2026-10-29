#!/usr/bin/env python3
"""
Deploy do Web App CEAP 3D para o Vercel via REST API.
Não requer npm ou Node.js instalados na máquina.
Faz upload direto dos arquivos para a CDN do Vercel e dispara o deployment.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_TEAM_ID = "team_230KK43BCazj8vZkadCGtT8X"
DEFAULT_TOKEN = "vcp_5ZfD1vUFEks5YXHY0h0MOk81hmwexfsBWjrWMzVaW94PM8xcsj3hIfd5"
DEFAULT_PROJECT = "ceap-latent-space-3d"
WEB_APP_DIR = Path("/home/lucas.ocunha/masters/web")
if not WEB_APP_DIR.exists():
    WEB_APP_DIR = Path("/home/lucas.ocunha/masters/web_app")


def compute_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def upload_file(team_id: str, token: str, file_path: Path, rel_path: str) -> dict:
    content = file_path.read_bytes()
    sha = compute_sha1(content)
    size = len(content)

    url = f"https://api.vercel.com/v2/files?teamId={team_id}"
    req = urllib.request.Request(
        url,
        data=content,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            "x-vercel-digest": sha,
            "Content-Length": str(size),
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return {"file": rel_path, "sha": sha, "size": size, "uploaded": True}
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        # Se retornar 409 ou similar indicando que o arquivo já existe, é sucesso
        if e.code in (200, 201, 409):
            return {"file": rel_path, "sha": sha, "size": size, "uploaded": False}
        print(f"Erro ao enviar {rel_path} (HTTP {e.code}): {body}")
        raise


def create_deployment(team_id: str, token: str, project_name: str, files_manifest: list) -> dict:
    url = f"https://api.vercel.com/v13/deployments?teamId={team_id}"
    payload = {
        "name": project_name,
        "files": files_manifest,
        "projectSettings": {
            "framework": None
        },
        "target": "production"
    }
    raw_body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=raw_body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Content-Length": str(len(raw_body))
        },
        method="POST"
    )

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def check_deployment_status(team_id: str, token: str, deployment_id: str) -> dict:
    url = f"https://api.vercel.com/v13/deployments/{deployment_id}?teamId={team_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())


def load_env():
    env_paths = [
        Path("/home/lucas.ocunha/masters/.env"),
        Path("/home/lucas.ocunha/masters/web_app/.env")
    ]
    for ep in env_paths:
        if ep.exists():
            for line in ep.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Deploy Web App CEAP 3D para o Vercel")
    parser.add_argument("--token", default=os.getenv("VERCEL_TOKEN", DEFAULT_TOKEN), help="Vercel Access Token")
    parser.add_argument("--team-id", default=os.getenv("VERCEL_TEAM_ID", DEFAULT_TEAM_ID), help="Vercel Team ID")
    parser.add_argument("--project", default=os.getenv("VERCEL_PROJECT", DEFAULT_PROJECT), help="Nome do projeto no Vercel")
    args = parser.parse_args()

    print("==================================================================")
    print("      DEPLOY CEAP 3D LATENT EXPLORER PARA O VERCEL               ")
    print("==================================================================")
    print(f"Diretório: {WEB_APP_DIR}")
    print(f"Team ID:   {args.team_id}")
    print(f"Projeto:   {args.project}")

    # 1. Coletar todos os arquivos em web_app/
    files_to_deploy = []
    for p in WEB_APP_DIR.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            rel = p.relative_to(WEB_APP_DIR).as_posix()
            files_to_deploy.append((p, rel))

    print(f"\n1. Encontrados {len(files_to_deploy)} arquivos para deploy.")

    # 2. Upload de cada arquivo para a CDN do Vercel
    manifest = []
    print("\n2. Enviando arquivos para a CDN do Vercel...")
    t_start_upload = time.time()
    for file_path, rel_path in files_to_deploy:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"   -> [{size_mb:6.2f} MB] {rel_path}...", end=" ", flush=True)
        res = upload_file(args.team_id, args.token, file_path, rel_path)
        manifest.append({
            "file": rel_path,
            "sha": res["sha"],
            "size": res["size"]
        })
        print(f"OK (sha: {res['sha'][:8]}...)")

    print(f"   Upload concluído em {time.time() - t_start_upload:.1f}s.")

    # 3. Criar Deployment na API do Vercel
    print(f"\n3. Disparando criação do deployment em '{args.project}'...")
    dep_data = create_deployment(args.team_id, args.token, args.project, manifest)

    dep_id = dep_data.get("id")
    dep_url = dep_data.get("url")
    readyState = dep_data.get("readyState") or dep_data.get("status")
    print(f"   Deployment ID: {dep_id}")
    print(f"   Preview URL:   https://{dep_url}")
    print(f"   Status Inicial: {readyState}")

    # 4. Aguardar status READY
    print("\n4. Aguardando finalização e publicação do build...")
    for _ in range(30):
        time.sleep(2)
        info = check_deployment_status(args.team_id, args.token, dep_id)
        st = info.get("readyState") or info.get("status")
        print(f"   Status: {st}...", flush=True)
        if st in ("READY", "ready"):
            print("\n==================================================================")
            print("                DEPLOY CONCLUÍDO COM SUCESSO!                     ")
            print("==================================================================")
            print(f"URL de Produção: https://{args.project}.vercel.app")
            print(f"URL do Deploy:   https://{dep_url}")
            print(f"Alias Oficial:   https://{info.get('alias', [dep_url])[0] if info.get('alias') else dep_url}")
            print("==================================================================")
            return
        elif st in ("ERROR", "CANCELED"):
            print(f"\nErro no deployment: {info.get('error')}")
            sys.exit(1)

    print(f"\nDeployment ainda em processamento: verifique em https://{dep_url}")


if __name__ == "__main__":
    main()
