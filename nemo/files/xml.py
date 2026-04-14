from pathlib import Path
from xml.etree import ElementTree as ET


def read_xml(file_path: str | Path) -> ET.Element:
    """
    Read an XML file and return its root element.

    Parameters
    ----------
    file_path : str | Path
        Path to the input XML file.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element of the parsed XML tree.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    xml.etree.ElementTree.ParseError
        If the XML content is malformed.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    tree = ET.parse(path)
    return tree.getroot()


def find_xml_elements(
    root: ET.Element,
    tag: str,
) -> list[ET.Element]:
    """
    Find all XML elements with the given tag under a root element.

    Parameters
    ----------
    root : ET.Element
        Root XML element.
    tag : str
        Tag name to search for.

    Returns
    -------
    list[ET.Element]
        List of matching XML elements.
    """
    return root.findall(f".//{tag}")


def find_xml_element(
    root: ET.Element,
    tag: str,
) -> ET.Element | None:
    """
    Find the first XML element with the given tag under a root element.

    Parameters
    ----------
    root : ET.Element
        Root XML element.
    tag : str
        Tag name to search for.

    Returns
    -------
    ET.Element | None
        First matching XML element, or ``None`` if not found.
    """
    return root.find(f".//{tag}")


def get_xml_element_text(
    element: ET.Element | None,
    default: str = "",
    strip: bool = True,
) -> str:
    """
    Get the text content of an XML element.

    Parameters
    ----------
    element : ET.Element | None
        XML element whose text content will be read.
    default : str, default=""
        Value returned when the element is ``None`` or has no text.
    strip : bool, default=True
        Whether to strip leading and trailing whitespace.

    Returns
    -------
    str
        Element text content, or ``default`` if unavailable.
    """
    if element is None or element.text is None:
        return default

    text = element.text
    return text.strip() if strip else text
