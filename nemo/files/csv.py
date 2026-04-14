from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal

import pandas as pd


def read_csv(
    file_path: str | Path,
    separator: str = ",",
    header: int | list[int] | None | Literal["infer"] = "infer",
    encoding: str = "utf-8",
    column_names: list[str] | None = None,
    dtypes: dict[str, Any] | None = None,
    skip_rows: int | list[int] | Callable[[int], bool] | None = None,
) -> pd.DataFrame:
    """
    Read a tabular text file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str | Path
        Path to the input file.
    sep : str, default=","
        Column separator used in the file.
    header : int, list of int, None, "infer", default="infer"
        Row number(s) to use as the column names, and the start of the
        data. Use ``None`` if the file does not contain a header row.
        Use ``"infer"`` to let pandas infer
        the header behavior.
    encoding : str, default="utf-8"
        File encoding.
    column_names : list of str | None, default=None
        Column names to assign to the resulting DataFrame. If provided,
        these names override the file header behavior.
    dtypes : dict of str to Any | None, default=None
        Optional mapping of column names to data types.
    skip_rows : int, list of int, callable, optional
        Line numbers to skip (0-indexed) or number of lines to skip (int)
        at the start of the file.

    Returns
    -------
    pd.DataFrame
        Loaded tabular data.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    ValueError
        If ``skip_rows`` is a negative integer.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if isinstance(skip_rows, int) and skip_rows < 0:
        raise ValueError("skip_rows cannot be a negative integer.")

    return pd.read_csv(
        path,
        sep=separator,
        header=header,
        encoding=encoding,
        names=column_names,
        dtype=dtypes,  # type: ignore
        skiprows=skip_rows,
    )


def write_csv(
    df: pd.DataFrame,
    file_path: str | Path,
    separator: str = ",",
    encoding: str = "utf-8",
    include_header: bool = True,
    include_index: bool = False,
    mode: Literal["w", "a", "x"] = "w",
) -> Path:
    """
    Write a pandas DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be written.
    file_path : str | Path
        Path to the output file.
    separator : str, default=","
        Column separator used in the output file.
    encoding : str, default="utf-8"
        File encoding.
    include_header : bool, default=True
        Whether to write the header row.
    include_index : bool, default=False
        Whether to write the DataFrame index.
    mode : {"w", "a", "x"}, default="w"
        File writing mode:
        - ``"w"`` overwrites the file
        - ``"a"`` appends to the file
        - ``"x"`` creates a new file and fails if it already exists

    Returns
    -------
    Path
        Path to the written file.

    Raises
    ------
    ValueError
        If ``separator`` is an empty string.
    """
    path = Path(file_path)

    if separator == "":
        raise ValueError("separator cannot be an empty string.")

    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        sep=separator,
        encoding=encoding,
        header=include_header,
        index=include_index,
        mode=mode,
    )

    return path
