import pytest

from nemo.retrieval_assignment.inverted_list import InvertedListGeneratorConfig
from nemo.retrieval_assignment.inverted_list import gen_inverted_list


@pytest.mark.slow
def test_gen_inverted_list():
    config = InvertedListGeneratorConfig.create()

    df = gen_inverted_list(
        file_paths=config.read_paths,
        # output_path=config.write_path,
    )

    assert df is not None


# pytest -k test_gen_inverted_list -s --log-cli-level=INFO
# pytest -k test_gen_inverted_list -s --log-cli-level=DEBUG


# pytest -k test_gen_inverted_list -s \
#   --log-cli-level=INFO \
#   --log-cli-format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s" \
#   --log-cli-date-format="%H:%M:%S"
