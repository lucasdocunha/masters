from pathlib import Path
from scripts.download import get_json_zip_urls
from scripts.download_pdfs import get_pdf_urls
from scripts.download_url import URL_TEMPLATE


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


if __name__ == "__main__":
    test_get_json_zip_urls()
    test_download_url_template()
    test_get_pdf_urls()
    print("OK")


