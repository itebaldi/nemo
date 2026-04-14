import logging
import time

from nemo.files.csv import read_csv
from nemo.files.xml import read_xml
from nemo.retrieval_assignment.inverted_list import InvertedListGeneratorConfig
from nemo.retrieval_assignment.inverted_list import gen_inverted_list
from nemo.retrieval_assignment.query_processor import QueryProcessorConfig
from nemo.retrieval_assignment.query_processor import gen_expected_docs
from nemo.retrieval_assignment.query_processor import gen_processed_queries
from nemo.retrieval_assignment.search_engine import SearcherConfig
from nemo.retrieval_assignment.search_engine import gen_results
from nemo.retrieval_assignment.vector_model import VectorModelConfig
from nemo.retrieval_assignment.vector_model import gen_vector_model
from nemo.vector_retrieval.tf_idf import VectorModel

logger = logging.getLogger(__name__)
WRITE_FILES = False

if __name__ == "__main__":
    # python -m nemo.vector_retrieval.main
    start = time.perf_counter()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Initializing query processor...")

    query_config = QueryProcessorConfig.create()
    root = read_xml(query_config.read_path)
    consultas_df = gen_processed_queries(
        xml_root=root,
        output_path=query_config.queries_output_path if WRITE_FILES else None,
    )

    esperados_df = gen_expected_docs(
        xml_root=root,
        output_path=query_config.expected_output_path if WRITE_FILES else None,
    )

    logger.info("Initializing inverted list...")

    inverted_list_config = InvertedListGeneratorConfig.create()
    lista_invertida_df = gen_inverted_list(
        file_paths=inverted_list_config.read_paths,
        output_path=inverted_list_config.write_path if WRITE_FILES else None,
    )

    logger.info("Initializing vector model...")

    vector_config = VectorModelConfig.create()
    vector_model_df = gen_vector_model(
        read_path=vector_config.read_path,
        output_path=vector_config.write_path if WRITE_FILES else None,
    )

    logger.info("Initializing search engine...")

    search_config = SearcherConfig.create()

    if WRITE_FILES:
        logger.info("Module 4 — load vector model from %s", search_config.model_path)
        vector_model_df = VectorModel.dataframe_from_csv(search_config.model_path)

        consultas_df = read_csv(
            search_config.queries_path,
            separator=";",
            dtypes={"QueryNumber": str, "QueryText": str},
        )

        logger.info(
            "Loaded %d queries from %s; ranking by cosine similarity",
            len(consultas_df),
            search_config.queries_path,
        )

    results_df = gen_results(
        vector_model=vector_model_df,
        queries=consultas_df,
        output_path=search_config.results_path if WRITE_FILES else None,
    )

    logger.info(
        "Finished project in %.3fs",
        time.perf_counter() - start,
    )
