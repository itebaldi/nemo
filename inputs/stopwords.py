from pathlib import Path

import nltk
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.corpus import stopwords

NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"
DEFAULT_RESOURCES: tuple[str, ...] = ("stopwords",)


def initialize_nltk() -> None:
    """
    Register the local NLTK data directory.

    Returns
    -------
    None
    """
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.append(str(NLTK_DATA_DIR))


def validate_nltk_resources(resources: tuple[str, ...] = DEFAULT_RESOURCES) -> None:
    """
    Validate that required NLTK resources are available locally.

    Parameters
    ----------
    resources : tuple[str, ...], default=("stopwords",)
        Names of NLTK resources expected in the local NLTK data directory.

    Returns
    -------
    None

    Raises
    ------
    LookupError
        If any required resource is not available.
    """
    initialize_nltk()

    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError as exc:
            raise LookupError(
                f"The NLTK resource '{resource}' was not found in the local "
                f"repository data directory: {NLTK_DATA_DIR}"
            ) from exc


def get_english_stopwords() -> set[str]:
    """
    Return the English stopword set from NLTK.

    Returns
    -------
    set[str]
        English stopwords.
    """
    validate_nltk_resources()
    return set(stopwords.words("english"))


def get_portuguese_stopwords() -> set[str]:
    """
    Return the Portuguese stopword set from NLTK.

    Returns
    -------
    set[str]
        Portuguese stopwords.
    """
    validate_nltk_resources()
    return set(stopwords.words("portuguese"))


def detect_language(text: str) -> str | None:
    """
    Detect the language of a text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str | None
        ISO language code such as ``"en"`` or ``"pt"``,
        or ``None`` if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return None


def get_stop_words_for_text(text: str) -> set[str]:
    """
    Get stop words according to the detected text language.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    set[str]
        Stop words for the detected language.
    """
    language = detect_language(text)

    if language == "pt":
        return get_portuguese_stopwords()

    if language == "en":
        return get_english_stopwords()

    return set()


if __name__ == "__main__":
    # python -m inputs.stopwords
    initialize_nltk()
