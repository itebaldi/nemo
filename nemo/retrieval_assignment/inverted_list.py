# módulo 2 - Gerador Lista Invertida


import logging
import time
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.files.csv import write_csv
from nemo.files.xml import find_xml_element
from nemo.files.xml import find_xml_elements
from nemo.files.xml import get_xml_element_text
from nemo.files.xml import read_xml
from nemo.vector_retrieval.indexing import Document
from nemo.vector_retrieval.indexing import InvertedIndex
from nemo.vector_retrieval.indexing import InvertedIndexMatrix
from nemo.vector_retrieval.indexing import gen_inverted_index

logger = logging.getLogger(__name__)


class InvertedListGeneratorConfig(BaseModel):
    """
    Configuration for the inverted list generator module.

    Attributes
    ----------
    read_paths : list[Path]
        Paths to the input XML files.
    write_path : Path
        Path to the inverted list CSV output file.
    """

    model_config = ConfigDict(frozen=True)

    read_paths: list[Path]
    write_path: Path

    @classmethod
    def create(
        cls,
        config_path: str | Path | None = None,
    ) -> "InvertedListGeneratorConfig":
        """
        Create an inverted list generator configuration from a CFG file.

        Parameters
        ----------
        config_path : str | Path | None, default=None
            Path to the configuration file. If ``None``, a default path
            is used.

        Returns
        -------
        InvertedListGeneratorConfig
            Parsed inverted list generator configuration.
        """
        logger.info("Reading inverted list config...")

        path = (
            Path(config_path)
            if config_path is not None
            else Path("inputs/vector_retrieval/GLI.CFG")
        )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        read_paths: list[Path] = []
        write_path: Path | None = None

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

            if key == "LEIA":
                read_paths.append(Path(value))
                continue

            if key == "ESCREVA":
                write_path = Path(value)
                continue

            raise ValueError(
                f"Unexpected config key '{key}' in {path}:{line_number}."
            )

        if not read_paths:
            raise ValueError("At least one LEIA instruction is required.")

        if write_path is None:
            raise ValueError("Missing required ESCREVA instruction.")

        resolved = cls(
            read_paths=read_paths,
            write_path=write_path,
        )
        logger.info(
            "Inverted list config (%s): %d LEIA file(s) → %s",
            path,
            len(resolved.read_paths),
            resolved.write_path,
        )
        for rp in resolved.read_paths:
            logger.debug("  LEIA %s", rp)
        return resolved


def gen_inverted_list(
    file_paths: list[Path],
    output_path: str | Path | None = None,
) -> InvertedIndexMatrix:
    """
    Generate an inverted list from one or more XML files.

    Parameters
    ----------
    file_paths : list[str | Path]
        Paths to the input XML files.
    output_path : str | Path
        Path to the output CSV file.

    Returns
    -------
    InvertedIndexMatrix
        DataFrame containing the inverted list.
    """
    start = time.perf_counter()
    logger.info(
        "Module 2 — build inverted list from %d XML file(s)",
        len(file_paths),
    )

    documents = _gen_documents(file_paths)
    inverted_index: InvertedIndex = gen_inverted_index(documents)

    df = inverted_index.to_dataframe()
    logger.info(
        "Inverted index: %d distinct terms, %d DataFrame rows",
        len(inverted_index.root),
        len(df.root),
    )

    if output_path:
        write_csv(
            df=df.root,
            file_path=output_path,
            separator=";",
        )
        logger.info("Wrote inverted list to %s", output_path)

    logger.info(
        "Module 2 (inverted list) finished in %.3fs",
        time.perf_counter() - start,
    )
    return df


def _gen_documents(file_paths: list[Path]) -> list[Document]:
    """
    Extract documents from the Cystic Fibrosis XML collection.

    Parameters
    ----------
    file_paths : list[str | Path]
        Paths to the input XML files.

    Returns
    -------
    list[Document]
        Extracted documents.
    """
    documents: list[Document] = []

    for file_path in file_paths:
        root = read_xml(file_path)
        records = find_xml_elements(root, "RECORD")
        n_before = len(documents)

        for record in records:
            record_number = get_xml_element_text(
                find_xml_element(record, "RECORDNUM")
            )

            if not record_number:
                raise ValueError("Missing RECORDNUM in record.")

            record_id = int(record_number)

            abstract = get_xml_element_text(find_xml_element(record, "ABSTRACT"))
            extract = get_xml_element_text(find_xml_element(record, "EXTRACT"))

            record_text = abstract or extract or ""

            if not record_text:
                continue

            documents.append(
                Document(
                    document_id=record_id,
                    text=record_text,
                )
            )

        added = len(documents) - n_before
        logger.debug("Extracted %d documents with text from %s", added, file_path)

    logger.info(
        "Documents with ABSTRACT or EXTRACT text: %d",
        len(documents),
    )

    return documents
