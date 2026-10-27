import json
from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from pydantic import ValidationError

from scripts.gemini_extractor import (
    ExtracaoDocumentoFiscal,
    extract_pdf_with_gemini,
    PROMPT_SISTEMA,
)


def test_extracao_documento_fiscal_schema_valid():
    """Testa se o schema Pydantic valida corretamente os dados fiscais."""
    data = {
        "fornecedor_nome": "POSTO IPIRANGA LTDA",
        "fornecedor_cnpj_cpf": "12.345.678/0001-90",
        "numero_documento": "001234",
        "data_emissao": "2024-05-10",
        "valor_total": 150.50,
        "descricao_resumida": "Abastecimento Gasolina Comum",
        "observacoes": "Cupom perfeitamente legível",
    }
    item = ExtracaoDocumentoFiscal(**data)
    dumped = item.model_dump()
    assert dumped["fornecedor_nome"] == "POSTO IPIRANGA LTDA"
    assert dumped["fornecedor_cnpj_cpf"] == "12.345.678/0001-90"
    assert dumped["numero_documento"] == "001234"
    assert dumped["data_emissao"] == "2024-05-10"
    assert dumped["valor_total"] == 150.50
    assert dumped["descricao_resumida"] == "Abastecimento Gasolina Comum"


def test_extracao_documento_fiscal_schema_defaults():
    """Testa se o schema aceita campos nulos/ausentes como esperado para documentos fiscais incompletos."""
    item = ExtracaoDocumentoFiscal()
    dumped = item.model_dump()
    assert dumped["fornecedor_nome"] is None
    assert dumped["fornecedor_cnpj_cpf"] is None
    assert dumped["valor_total"] is None
    assert dumped["data_emissao"] is None


def test_extracao_documento_fiscal_schema_type_conversion():
    """Testa conversão de tipos (ex: string numérica para float no valor total)."""
    item = ExtracaoDocumentoFiscal(valor_total="99.90")
    assert item.valor_total == 99.90

    with pytest.raises(ValidationError):
        ExtracaoDocumentoFiscal(valor_total="nao-numerico")


def test_extract_pdf_with_gemini_parsed(tmp_path: Path):
    """Testa extração com resposta já parseada pela API do Gemini."""
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

    mock_client = MagicMock()
    mock_model_parsed = ExtracaoDocumentoFiscal(
        fornecedor_nome="AUTO POSTO TESTE",
        fornecedor_cnpj_cpf="11222333000144",
        numero_documento="54321",
        data_emissao="2023-11-20",
        valor_total=250.0,
    )
    mock_response = MagicMock()
    mock_response.parsed = mock_model_parsed
    mock_client.models.generate_content.return_value = mock_response

    result = extract_pdf_with_gemini(mock_client, dummy_pdf, model_name="gemini-2.5-flash")
    assert result["fornecedor_nome"] == "AUTO POSTO TESTE"
    assert result["valor_total"] == 250.0
    assert result["numero_documento"] == "54321"
    mock_client.models.generate_content.assert_called_once()


def test_extract_pdf_with_gemini_fallback_json(tmp_path: Path):
    """Testa fallback quando parsed é nulo mas o texto da resposta é JSON válido."""
    dummy_pdf = tmp_path / "test.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.parsed = None
    mock_response.text = json.dumps({
        "fornecedor_nome": "RESTAURANTE FALLBACK",
        "valor_total": 75.0,
    })
    mock_client.models.generate_content.return_value = mock_response

    result = extract_pdf_with_gemini(mock_client, dummy_pdf)
    assert result["fornecedor_nome"] == "RESTAURANTE FALLBACK"
    assert result["valor_total"] == 75.0


def test_prompt_sistema_contains_required_fields():
    """Valida se o prompt do sistema instrui a extração dos 5 campos fiscais essenciais."""
    for field_kw in ["Fornecedor", "CNPJ", "Número", "Data de emissão", "Valor total"]:
        assert field_kw.lower() in PROMPT_SISTEMA.lower()
