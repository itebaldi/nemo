from pathlib import Path

from bs4 import BeautifulSoup


def read_html(path: str | Path) -> BeautifulSoup:
    """
    Read an HTML file and parse it into a BeautifulSoup object.

    Parameters
    ----------
    path : str | Path
        Path to the input HTML file.

    Returns
    -------
    BeautifulSoup
        Parsed HTML content as a BeautifulSoup object.
    """
    html_path = Path(path)
    html = html_path.read_text(encoding="utf-8")
    return BeautifulSoup(html, "html.parser")


def write_html(html: BeautifulSoup, output_path: str | Path) -> Path:
    """
    Write a BeautifulSoup object to an HTML file.

    Parameters
    ----------
    html : BeautifulSoup
        The BeautifulSoup object to be written to a file.
    output_path : str | Path
        Path to the output HTML file.

    Returns
    -------
    Path
        The path where the file was saved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(html), encoding="utf-8")
    return path
