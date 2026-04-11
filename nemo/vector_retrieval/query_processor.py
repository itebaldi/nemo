# módulo 1 - Processador de Consultas


from pathlib import Path

import pandas as pd
from toolz.functoolz import pipe

from nemo.importing import read_xml
from nemo.importing import write_csv
from nemo.preprocessing.text import remove_text_accents
from nemo.preprocessing.text import remove_text_punctuation
from nemo.preprocessing.text import replace_text_substrings
from nemo.preprocessing.text import uppercase_text
from nemo.preprocessing.xml import find_xml_element
from nemo.preprocessing.xml import find_xml_elements
from nemo.preprocessing.xml import get_xml_element_text


def gen_processed_queries(
    file_path: Path,
    output_path: Path | None,
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
    root = read_xml(file_path)
    queries = find_xml_elements(root, "QUERY")

    records: list[dict[str, str]] = []

    for query in queries[:2]:
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
    )


def gen_expected_docs(): ...
