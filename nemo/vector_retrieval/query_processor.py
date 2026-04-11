# módulo 1 - Processador de Consultas


from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from toolz.functoolz import pipe

from nemo.importing import read_xml
from nemo.importing import write_csv
from nemo.preprocessing.text import normalize_text_whitespace
from nemo.preprocessing.text import remove_text_accents
from nemo.preprocessing.text import remove_text_punctuation
from nemo.preprocessing.text import replace_text_substrings
from nemo.preprocessing.text import uppercase_text
from nemo.preprocessing.xml import find_xml_element
from nemo.preprocessing.xml import find_xml_elements
from nemo.preprocessing.xml import get_xml_element_text


class QueryProcessorConfig(BaseModel):
    """
    Configuration for the query processor module.

    Attributes
    ----------
    read_path : Path
        Path to the input XML file.
    queries_output_path : Path
        Path to the processed queries CSV output file.
    expected_output_path : Path
        Path to the expected documents CSV output file.
    """

    model_config = ConfigDict(frozen=True)

    read_path: Path
    queries_output_path: Path
    expected_output_path: Path

    @classmethod
    def create(cls, config_path: str | Path | None = None) -> "QueryProcessorConfig":
        """
        Read and validate the query processor configuration file.

        Parameters
        ----------
        file_path : str | Path
            Path to the ``PC.CFG`` file.

        Returns
        -------
        QueryProcessorConfig
            Parsed query processor configuration.
        """

        path = (
            Path(config_path)
            if config_path is not None
            else Path("inputs/vector_retrieval/PC.CFG")
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

        required_keys = {"LEIA", "CONSULTAS", "ESPERADOS"}
        missing_keys = required_keys - set(config)

        if missing_keys:
            raise ValueError(f"Missing required config keys: {sorted(missing_keys)}")

        return cls(
            read_path=Path(config["LEIA"]),
            queries_output_path=Path(config["CONSULTAS"]),
            expected_output_path=Path(config["ESPERADOS"]),
        )


def gen_processed_queries(
    xml_root: ET.Element,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a processed queries CSV from the input XML file.

    Parameters
    ----------
    file_path : Path
        Path to the input XML file containing the queries.
    output_path : Path
        Path to the output CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the processed queries.

    Notes
    -----
    The generated CSV uses ``;`` as separator and includes the columns
    ``QueryNumber`` and ``QueryText``.
    """
    queries = find_xml_elements(xml_root, "QUERY")

    records: list[dict[str, str]] = []

    for query in queries:
        query_number = get_xml_element_text(find_xml_element(query, "QueryNumber"))
        query_text = get_xml_element_text(find_xml_element(query, "QueryText"))

        processed_query_text = _normalize_text(query_text)

        records.append(
            {
                "QueryNumber": query_number,
                "QueryText": processed_query_text,
            }
        )

    df = pd.DataFrame(records)

    if output_path:
        write_csv(
            df=df,
            file_path=output_path,
            separator=";",
        )

    return df


def _normalize_text(text: str) -> str:

    return pipe(
        replace_text_substrings(text, ";", ""),
        remove_text_punctuation,
        uppercase_text,
        remove_text_accents,
        normalize_text_whitespace,
    )


def gen_expected_docs(
    xml_root: ET.Element,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate an expected documents CSV from the input XML file.

    Parameters
    ----------
    file_path : str | Path
        Path to the input XML file containing the queries.
    output_path : str | Path
        Path to the output CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the expected documents for each query.

    Notes
    -----
    The generated CSV uses ``;`` as separator and includes the columns
    ``QueryNumber``, ``DocNumber`` and ``DocVotes``.
    """
    queries = find_xml_elements(xml_root, "QUERY")

    records: list[dict[str, str | int]] = []

    for query in queries:
        query_number = get_xml_element_text(find_xml_element(query, "QueryNumber"))
        records_element = find_xml_element(query, "Records")

        if records_element is None:
            continue

        items = find_xml_elements(records_element, "Item")

        for item in items:
            doc_number = get_xml_element_text(item)
            score = item.attrib.get("score", "")
            doc_votes = sum(character != "0" for character in score)

            records.append(
                {
                    "QueryNumber": query_number,
                    "DocNumber": doc_number,
                    "DocVotes": doc_votes,
                }
            )

    df = pd.DataFrame(records)

    if output_path:
        write_csv(
            df=df,
            file_path=output_path,
            separator=";",
        )

    return df
