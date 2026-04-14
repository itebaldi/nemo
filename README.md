# Nemo

Repositório organizado como **biblioteca Python** de ferramentas para **mineração e processamento de texto**: leitura de CSV/XML, pré-processamento, indexação, modelo vetorial (TF-IDF) e experimentos em torno de aplicações.

Há **testes unitários** em `tests/` (pytest) para manter o comportamento dos módulos estável à medida que o código evolui.

## Requisitos

- Python **3.11** (intervalo definido em `pyproject.toml`)
- Dependências gerenciadas com [Poetry](https://python-poetry.org/)

## Configuração do ambiente

Na raiz do repositório:

```sh
conda create --prefix ./venv python=3.11 -y
conda activate ./venv
poetry env use "$(which python)"
poetry install
```

## Testes

```sh
pytest
```

## Sistema de recuperação em memória (modelo vetorial)

Implementação em pipeline, em linha com trabalhos clássicos de RI: processamento de consultas a partir de XML, geração de lista invertida, construção do **modelo vetorial TF-IDF** e ranqueamento por similaridade de cosseno.

### Como executar

Na raiz do repositório (com o ambiente ativado e dependências instaladas):

```sh
python -m nemo.retrieval_assignment.main
```

Por padrão, o fluxo roda **em memória** (sem gravar CSVs intermediários). Os caminhos de leitura e escrita vêm dos arquivos `.CFG` em `inputs/vector_retrieval/`.

### Arquivos de configuração

| Arquivo | Chaves principais | Função |
|---------|-------------------|--------|
| `inputs/vector_retrieval/PC.CFG` | `LEIA`, `CONSULTAS`, `ESPERADOS` | Módulo 1 — processador de consultas (XML → CSV) |
| `inputs/vector_retrieval/GLI.CFG` | `LEIA`, `ESCREVA` | Módulo 2 — gerador de lista invertida |
| `inputs/vector_retrieval/INDEX.CFG` | `LEIA`, `ESCREVA` | Módulo 3 — indexador (lista invertida → modelo vetorial) |
| `inputs/vector_retrieval/BUSCA.CFG` | `MODELO`, `CONSULTAS`, `RESULTADOS` | Módulo 4 — buscador |

Cada linha útil nos `.CFG` segue o formato `CHAVE=caminho`.

### Pasta de resultados

As saídas configuradas nos `.CFG` apontam, neste repositório, para **`outputs/vector_retrieval/RESULT/`** (a pasta é criada na gravação, se ainda não existir). Arquivos típicos:

| Arquivo | Origem (CFG / módulo) |
|---------|------------------------|
| `consultas.csv` | `PC.CFG` — consultas processadas |
| `esperados.csv` | `PC.CFG` — documentos esperados por consulta |
| `lista_invertida.csv` | `GLI.CFG` — lista invertida |
| `modelo_vetorial.csv` | `INDEX.CFG` — modelo vetorial TF-IDF |
| `resultados.csv` | `BUSCA.CFG` — ranqueamento das consultas |

Com **`WRITE_FILES = False`** em `nemo/retrieval_assignment/main.py`, o `main` só processa em memória e **não** grava esses CSVs. Defina `WRITE_FILES = True` para gerar os arquivos nessa pasta (e, no fluxo completo em disco, o buscador recarrega modelo e consultas a partir dos caminhos de `BUSCA.CFG`).

### Código dos módulos

- `nemo/retrieval_assignment/query_processor.py` — consultas e documentos esperados
- `nemo/retrieval_assignment/inverted_list.py` — lista invertida
- `nemo/retrieval_assignment/vector_model.py` — matriz TF-IDF
- `nemo/retrieval_assignment/search_engine.py` — ranqueamento
- `nemo/vector_retrieval/` — núcleo do modelo vetorial e do índice invertido

### Formato do arquivo de modelo (MODELO)

A descrição detalhada do CSV do modelo vetorial (colunas, separador, semântica dos pesos) está em **`MODELO.TXT`** na raiz do repositório.

