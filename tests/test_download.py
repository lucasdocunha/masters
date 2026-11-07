from pathlib import Path
from dados.ceap.download import get_json_zip_urls
from dados.ceap.download_pdfs import DEFAULT_DEST, download_pdf, get_pdf_urls
from dados.ceap.download_url import DEFAULT_DEST as METADATA_DEST, URL_TEMPLATE
from dados.config import PDF_DIR, RAW_DIR


def test_get_json_zip_urls() -> None:
    urls = get_json_zip_urls(2014)
    assert len(urls) > 0
    assert all("cotas/Ano-" in u and u.endswith(".json.zip") for u in urls)
    assert any("Ano-2014" in u for u in urls)


def test_download_url_template() -> None:
    url = URL_TEMPLATE.format(year=2024)
    assert url == "https://www.camara.leg.br/cotas/Ano-2024.json.zip"


def test_get_pdf_urls() -> None:
    path = Path("Ano-2008.json")
    if path.exists():
        urls = get_pdf_urls(path)
        assert isinstance(urls, list)


def test_default_paths_come_from_config() -> None:
    assert DEFAULT_DEST == PDF_DIR
    assert METADATA_DEST == RAW_DIR


def test_download_pdf_reports_each_outcome(tmp_path: Path) -> None:
    source = tmp_path / "origem.pdf"
    source.write_bytes(b"%PDF")
    dest = tmp_path / "destino"
    dest.mkdir()

    assert download_pdf(source.as_uri(), dest) == "baixado"
    assert (dest / "origem.pdf").read_bytes() == b"%PDF"
    assert download_pdf(source.as_uri(), dest) == "existente"
    assert download_pdf((tmp_path / "nao_existe.pdf").as_uri(), dest) == "falha"
    assert not list(dest.glob("*.tmp"))


if __name__ == "__main__":
    test_get_json_zip_urls()
    test_download_url_template()
    test_get_pdf_urls()
    print("OK")


