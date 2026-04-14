# módulo 3 - Indexador

import logging
import time
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.files.csv import write_csv
from nemo.vector_retrieval.indexing import InvertedIndex
from nemo.vector_retrieval.tf_idf import VectorModel
from nemo.vector_retrieval.tf_idf import gen_vector_space_model

logger = logging.getLogger(__name__)


class VectorModelConfig(BaseModel):
    """
    Configuration for the vector model module.

    Attributes
    ----------
    read_path : Path
        Path to the input CSV file.
    write_path : Path
        Path to the write CSV output file.
    """

    model_config = ConfigDict(frozen=True)

    read_path: Path
    write_path: Path

    @classmethod
    def create(cls, config_path: str | Path | None = None) -> "VectorModelConfig":
        """
        Read and validate the vector model configuration file.

        Parameters
        ----------
        file_path : str | Path
            Path to the ``INDEX.CFG`` file.

        Returns
        -------
        VectorModelConfig
            Vector model configuration.
        """

        logger.info("Reading vector model config...")

        path = (
            Path(config_path)
            if config_path is not None
            else Path("inputs/vector_retrieval/INDEX.CFG")
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

        required_keys = {"LEIA", "ESCREVA"}
        missing_keys = required_keys - set(config)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {sorted(missing_keys)}")

        resolved = cls(
            read_path=Path(config["LEIA"]),
            write_path=Path(config["ESCREVA"]),
        )
        logger.info(
            "Indexer config (%s): LEIA=%s ESCREVA=%s",
            path,
            resolved.read_path,
            resolved.write_path,
        )
        return resolved


def gen_vector_model(
    read_path: Path,
    output_path: str | Path | None = None,
) -> VectorModel:

    start = time.perf_counter()
    logger.info("Module 3 — load inverted list from %s", read_path)
    inverted_index = InvertedIndex.dataframe_from_csv(file_path=read_path)
    logger.info("Terms in inverted list: %d", len(inverted_index.root))

    logger.info(
        "Building vector model (per-document normalized TF + standard IDF)",
    )
    vector_model = gen_vector_space_model(
        inverted_index=inverted_index,
        tf_method=VectorModel.normalized_tf_method,
        idf_method=VectorModel.standard_idf_method,
    )
    n_docs = len(vector_model.root.columns)
    logger.info(
        "Vector model matrix: %d terms × %d documents",
        len(vector_model.root),
        n_docs,
    )

    if output_path:
        write_csv(
            df=vector_model.root,
            file_path=output_path,
            separator=";",
            include_index=True,
        )
        logger.info("Wrote vector model to %s", output_path)

    logger.info(
        "Module 3 (vector model) finished in %.3fs",
        time.perf_counter() - start,
    )
    return vector_model
