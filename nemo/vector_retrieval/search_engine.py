# módulo 4 - Buscador


from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.importing import read_csv
from nemo.importing import write_csv
from nemo.preprocessing.query import Query
from nemo.preprocessing.tf_idf import VectorModel


class SearcherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: Path
    queries_path: Path
    results_path: Path

    @classmethod
    def create(cls, config_path: str | Path | None = None) -> "SearcherConfig":

        path = (
            Path(config_path)
            if config_path is not None
            else Path("inputs/vector_retrieval/BUSCA.CFG")
        )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        config: dict[str, str] = {}

        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            if "=" not in line:
                raise ValueError(
                    f"Invalid config line at {path}:{line_number}. "
                    "Expected format KEY=VALUE."
                )

            key, value = line.split("=", maxsplit=1)
            key = key.strip()
            value = value.strip()

            config[key] = value

        required_keys = {"MODELO", "CONSULTAS", "RESULTADOS"}
        missing_keys = required_keys - set(config)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {sorted(missing_keys)}")

        return cls(
            model_path=Path(config["MODELO"]),
            queries_path=Path(config["CONSULTAS"]),
            results_path=Path(config["RESULTADOS"]),
        )


def gen_results(write_output: bool = False) -> pd.DataFrame:

    config = SearcherConfig.create()

    vector_model = VectorModel.dataframe_from_csv(config.model_path)

    queries = read_csv(
        config.queries_path,
        separator=";",
        dtypes={"QueryNumber": str, "QueryText": str},
    )

    rows: list[dict[str, object]] = []

    for query in _queries_from_dataframe(queries):
        ranked_documents = query.search(vector_model)

        results = [
            (
                ranked_document.rank,
                ranked_document.document_id,
                ranked_document.score,
            )
            for ranked_document in ranked_documents
        ]

        rows.append(
            {
                "QueryNumber": query.query_id,
                "Results": results,
            }
        )
    df = pd.DataFrame(rows)

    if write_output:
        write_csv(
            df=df,
            file_path=config.results_path,
            separator=";",
        )

    return df


def _queries_from_dataframe(df: pd.DataFrame) -> list[Query]:
    """
    Create queries from a processed queries DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``QueryNumber`` and ``QueryText``.

    Returns
    -------
    list[Query]
        Parsed queries.
    """
    return [
        Query(
            query_id=str(row.QueryNumber),
            text=str(row.QueryText),
        )
        for row in df.itertuples(index=False)
    ]
