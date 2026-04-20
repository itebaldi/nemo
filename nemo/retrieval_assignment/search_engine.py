# módulo 4 - Buscador


import logging
import time
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.files.csv import write_csv
from nemo.vector_retrieval.query import Query
from nemo.vector_retrieval.search import SearchResults
from nemo.vector_retrieval.search import search
from nemo.vector_retrieval.tf_idf import VectorModel

logger = logging.getLogger(__name__)


class SearcherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_path: Path
    queries_path: Path
    results_path: Path

    @classmethod
    def create(cls, config_path: str | Path | None = None) -> "SearcherConfig":

        logger.info("Reading search engine config...")

        path = (
            Path(config_path)
            if config_path is not None
            else Path("inputs/vector_retrieval/BUSCA.CFG")
        )

        if not path.exists():
            # logger.error(f"Error opening CSV file: {e}")
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

        resolved = cls(
            model_path=Path(config["MODELO"]),
            queries_path=Path(config["CONSULTAS"]),
            results_path=Path(config["RESULTADOS"]),
        )
        logger.info(
            "Searcher config (%s): MODELO=%s CONSULTAS=%s RESULTADOS=%s",
            path,
            resolved.model_path,
            resolved.queries_path,
            resolved.results_path,
        )
        return resolved


def gen_results(
    vector_model: VectorModel,
    queries: DataFrame,
    output_path: str | Path | None = None,
) -> SearchResults:

    start = time.perf_counter()

    n_docs = len(vector_model.root.columns)
    logger.info("Documents in model: %d", n_docs)

    queries_df = _queries_from_dataframe(queries)
    search_results = search(queries_df, vector_model)

    df = search_results.to_dataframe()
    logger.info("Search finished: %d queries ranked", len(df))

    if output_path:
        write_csv(
            df=df,
            file_path=output_path,
            separator=";",
        )
        logger.info("Wrote search results to %s", output_path)
    else:
        logger.debug("Results kept in memory only (output_path=None)")

    logger.info(
        "Module 4 (search) finished in %.3fs",
        time.perf_counter() - start,
    )
    return search_results


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
