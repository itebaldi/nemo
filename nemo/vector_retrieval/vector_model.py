# módulo 3 - Indexador

from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict

from nemo.importing import write_csv
from nemo.preprocessing.indexing import InvertedIndex
from nemo.preprocessing.tf_idf import VectorModel
from nemo.preprocessing.tf_idf import gen_vector_space_model


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

        return cls(
            read_path=Path(config["LEIA"]),
            write_path=Path(config["ESCREVA"]),
        )


def gen_vector_model(
    read_path: Path,
    output_path: str | Path | None = None,
) -> VectorModel:

    inverted_index = InvertedIndex.dataframe_from_csv(file_path=read_path)

    vector_model = gen_vector_space_model(
        inverted_index=inverted_index,
        tf_method=VectorModel.normalized_tf_method,
        idf_method=VectorModel.standard_idf_method,
    )

    if output_path:
        write_csv(
            df=vector_model.root,
            file_path=output_path,
            separator=";",
            include_index=True,
        )

    return vector_model
