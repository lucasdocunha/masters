# Relatório do Dataset de Comprovantes CEAP (2014–2026)

**Data da geração:** 20 de outubro de 2026  
**Local dos arquivos salvos:** `/media/ssd2/datasets/CEAP/`  
**Origem dos metadados:** `data/raw/Ano-*.json` (Câmara dos Deputados)  
**Script de download:** `scripts/download_pdfs.py`

---

## 1. Visão Geral Consolidada

| Métrica | Valor Consolidado |
| :--- | :--- |
| **Período analisado** | 2014 a 2026 (13 anos) |
| **Total de PDFs esperados** | **1.406.013** |
| **Total de PDFs baixados com sucesso** | **1.402.334** (**99,74%**) |
| **Total de falhas no download (ausentes)** | **3.679** (**0,26%**) |
| **Espaço total em disco ocupado** | ~**266 GB** |
| **Páginas por documento** | Mín: **0** \| Média: **1,49** \| Mediana: **1** \| Máx: **88** |
| **Tamanho por documento** | Mín: **15 B** \| Média: **202,3 KB** \| Mediana: **122,2 KB** \| Máx: **1,37 MB** |
| **PDFs válidos e renderizáveis (OK)** | **1.402.313** (**99,9985%** dos baixados) |
| **PDFs protegidos por senha** | **1** (< 0,0001%) |
| **PDFs quebrados / corrompidos** | **10** (0,0007%) |
| **Arquivos inválidos (não são PDF)** | **10** (0,0007%) |
| **Falha de renderização (render fault)** | **0** (0%) |

---

## 2. Quantidade de Arquivos e Falhas por Ano

| Ano | Total Esperado | Baixados | Falhas | Taxa de Falha | Taxa de Sucesso |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2014** | 73.376 | 73.326 | 50 | 0,07% | 99,93% |
| **2015** | 189.280 | 188.539 | 741 | 0,39% | 99,61% |
| **2016** | 182.161 | 182.067 | 94 | 0,05% | 99,95% |
| **2017** | 174.326 | 172.223 | 2.103 | 1,21% | 98,79% |
| **2018** | 87.994 | 87.938 | 56 | 0,06% | 99,94% |
| **2019** | 105.256 | 105.194 | 62 | 0,06% | 99,94% |
| **2020** | 77.937 | 77.856 | 81 | 0,10% | 99,90% |
| **2021** | 91.160 | 91.040 | 120 | 0,13% | 99,87% |
| **2022** | 82.768 | 82.696 | 72 | 0,09% | 99,91% |
| **2023** | 91.703 | 91.654 | 49 | 0,05% | 99,95% |
| **2024** | 95.780 | 95.649 | 131 | 0,14% | 99,86% |
| **2025** | 91.583 | 91.509 | 74 | 0,08% | 99,92% |
| **2026** | 62.689 | 62.643 | 46 | 0,07% | 99,93% |
| **TOTAL** | **1.406.013** | **1.402.334** | **3.679** | **0,26%** | **99,74%** |

---

## 3. Integridade e Renderização dos PDFs Baixados

Todos os 1.402.334 arquivos salvos foram verificados individualmente quanto à validade do cabeçalho PDF, proteção por senha, integridade estrutural das páginas e capacidade de renderização dos operadores visuais.

| Ano | Baixados | Válidos & Renderizáveis (OK) | Com Senha | Quebrados / Corrompidos | Inválidos (Não-PDF) | Falha de Renderização |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2014** | 73.326 | 73.326 (100%) | 0 | 0 | 0 | 0 |
| **2015** | 188.539 | 188.539 (100%) | 0 | 0 | 0 | 0 |
| **2016** | 182.067 | 182.064 (99,998%) | 0 | 1 | 2 | 0 |
| **2017** | 172.223 | 172.216 (99,996%) | 0 | 2 | 5 | 0 |
| **2018** | 87.938 | 87.935 (99,997%) | 0 | 3 | 0 | 0 |
| **2019** | 105.194 | 105.189 (99,995%) | 1 | 1 | 3 | 0 |
| **2020** | 77.856 | 77.856 (100%) | 0 | 0 | 0 | 0 |
| **2021** | 91.040 | 91.040 (100%) | 0 | 0 | 0 | 0 |
| **2022** | 82.696 | 82.696 (100%) | 0 | 0 | 0 | 0 |
| **2023** | 91.654 | 91.653 (99,999%) | 0 | 1 | 0 | 0 |
| **2024** | 95.649 | 95.649 (100%) | 0 | 0 | 0 | 0 |
| **2025** | 91.509 | 91.509 (100%) | 0 | 0 | 0 | 0 |
| **2026** | 62.643 | 62.641 (99,997%) | 0 | 2 | 0 | 0 |
| **TOTAL** | **1.402.334** | **1.402.313 (99,9985%)** | **1** | **10** | **10** | **0** |

### Lista Completa dos Arquivos com Anomalias (21 arquivos no total)

| Ano | Arquivo | Problema | Detalhe |
| :---: | :---: | :---: | :--- |
| **2016** | `6005031.pdf` | Quebrado | Documento com 0 páginas |
| **2016** | `6090542.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2016** | `6164271.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2017** | `6220418.pdf` | Quebrado | Documento com 0 páginas |
| **2017** | `6369489.pdf` | Quebrado | Documento com 0 páginas |
| **2017** | `6296347.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2017** | `6434159.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2017** | `6458963.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2017** | `6472656.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2017** | `6472662.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2018** | `6571197.pdf` | Quebrado | Erro ao abrir estrutura PDF |
| **2018** | `6610288.pdf` | Quebrado | Documento com 0 páginas |
| **2018** | `6624260.pdf` | Quebrado | Documento com 0 páginas |
| **2019** | `6779152.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2019** | `6799101.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2019** | `6853779.pdf` | Quebrado | Erro ao abrir estrutura PDF |
| **2019** | `6946208.pdf` | Com Senha | Documento criptografado / requer senha para leitura |
| **2019** | `6967554.pdf` | Inválido | Não é PDF (apenas trailer sem cabeçalho `%PDF-`) |
| **2023** | `7629617.pdf` | Quebrado | Documento com 0 páginas |
| **2026** | `8075702.pdf` | Quebrado | Documento com 0 páginas |
| **2026** | `8075705.pdf` | Quebrado | Documento com 0 páginas |

---

## 4. Estatísticas de Número de Páginas

| Ano | Mínimo | Média | Mediana | Máximo |
| :---: | :---: | :---: | :---: | :---: |
| **2014** | 1 | 1,18 | 1 | 17 |
| **2015** | 1 | 1,26 | 1 | 68 |
| **2016** | 0 | 1,33 | 1 | 40 |
| **2017** | 0 | 1,46 | 1 | 36 |
| **2018** | 0 | 1,29 | 1 | 23 |
| **2019** | 1 | 1,52 | 1 | 75 |
| **2020** | 1 | 1,58 | 1 | 88 |
| **2021** | 1 | 1,65 | 1 | 33 |
| **2022** | 1 | 1,63 | 1 | 44 |
| **2023** | 0 | 1,64 | 1 | 56 |
| **2024** | 1 | 1,68 | 1 | 50 |
| **2025** | 1 | 1,71 | 1 | 58 |
| **2026** | 0 | 2,05 | 1 | 70 |
| **GERAL** | **0** | **1,49** | **1** | **88** |

---

## 5. Estatísticas de Tamanho de Arquivo

| Ano | Mínimo | Média | Mediana | Máximo |
| :---: | :---: | :---: | :---: | :---: |
| **2014** | 5,8 KB | 119,1 KB | 62,4 KB | 1,00 MB |
| **2015** | 2,2 KB | 125,5 KB | 64,2 KB | 1,00 MB |
| **2016** | 71 B | 130,0 KB | 69,9 KB | 1,00 MB |
| **2017** | 70 B | 156,8 KB | 88,2 KB | 1,00 MB |
| **2018** | 15 B | 161,5 KB | 91,1 KB | 1,00 MB |
| **2019** | 15 B | 206,2 KB | 137,1 KB | 1,00 MB |
| **2020** | 1,2 KB | 240,9 KB | 164,7 KB | 1,00 MB |
| **2021** | 1,2 KB | 269,7 KB | 196,1 KB | 1,00 MB |
| **2022** | 1,3 KB | 294,0 KB | 235,5 KB | 1,37 MB |
| **2023** | 1,5 KB | 278,6 KB | 224,4 KB | 1,00 MB |
| **2024** | 1,5 KB | 282,5 KB | 234,8 KB | 1,00 MB |
| **2025** | 1,2 KB | 284,3 KB | 224,5 KB | 1,00 MB |
| **2026** | 3,2 KB | 295,3 KB | 218,9 KB | 1,00 MB |
| **GERAL** | **15 B** | **202,3 KB** | **122,2 KB** | **1,37 MB** |

---

## 6. Observações Técnicas

1. **Altíssimo Índice de Integridade:** Dos 1.402.334 arquivos baixados, **1.402.313 (99,9985%)** abriram e renderizaram sem qualquer erro ou restrição.
2. **Quase Inexistência de Arquivos com Senha:** Apenas 1 único documento em todo o histórico de 13 anos está protegido por senha (`6946208.pdf`).
3. **Predomínio de Página Única:** Em todos os anos a mediana é de exatamente **1 página** (cupons fiscais, recibos, DANFEs e passagens aéreas).
4. **Evolução do Tamanho dos Arquivos:** A mediana de tamanho saltou de ~62 KB em 2014 para ~220–235 KB nos anos recentes, devido à digitalização em maior resolução e notas fiscais eletrônicas modernas.
5. **Arquivos Inválidos/Truncados:** Os 10 arquivos categorizados como inválidos contêm apenas o trailer do PDF sem a árvore de objetos e o cabeçalho `%PDF-`, fruto de transmissões incompletas na base original da Câmara.
