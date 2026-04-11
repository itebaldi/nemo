# módulo 2 - Gerador Lista Invertida e 3 - Indexador


from collections import defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from toolz.functoolz import pipe

from nemo.importing import read_xml
from nemo.importing import write_csv
from nemo.preprocessing.indexing import Document
from nemo.preprocessing.indexing import gen_inverted_index
from nemo.preprocessing.xml import find_xml_element
from nemo.preprocessing.xml import find_xml_elements
from nemo.preprocessing.xml import get_xml_element_text


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

        return cls(
            read_paths=read_paths,
            write_path=write_path,
        )


def gen_inverted_list(
    file_paths: list[Path],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
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
    pd.DataFrame
        DataFrame containing the inverted list.
    """
    inverted_index: dict[str, list[int]] = defaultdict(list)

    inverted_index = pipe(
        _gen_documents(file_paths),
        gen_inverted_index,
    )  # type: ignore

    rows = [
        {
            "Word": term,
            "Documents": document_ids,
        }
        for term, document_ids in inverted_index.items()
    ]

    df = pd.DataFrame(rows)

    if output_path:
        write_csv(
            df=df,
            file_path=output_path,
            separator=";",
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

    return documents
