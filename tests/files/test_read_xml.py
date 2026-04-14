from pathlib import Path

from nemo.files.xml import read_xml


def test_read_xml():
    xml_path = Path("inputs/vector_retrieval/CysticFibrosis2/cfquery.xml")

    root = read_xml(xml_path)

    assert root.tag == "FILEQUERY"
