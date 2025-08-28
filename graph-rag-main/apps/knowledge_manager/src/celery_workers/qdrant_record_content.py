import json
import uuid
from typing import Dict

from .celery_app import celery_app
from apps.knowledge_manager.src.qdrant_engine.qdrant_processor import NeuralSearcher
from libs.python.schemas.configuration import Config
from libs.python.utils.logger import logger
from libs.python.utils.token_managment import TextPreProcessor


@celery_app.task()
def process_record_content(
    text_sample_knowledge_content: str,
    src_sample_knowledge_content: str,
    config: Dict,
):
    try:
        logger.info(
            f"[process_record_content] src_sample_knowledge_content={src_sample_knowledge_content};\n"
            f"CFG: {json.dumps(config, indent=2)}"
        )
        config = Config(**config)
        chunk_size = config.config_knowledge_manager.config_retrieval_mode.chunk_size
        embedder_model = (
            config.config_knowledge_manager.config_retrieval_mode.embedder_model
        )
        collection_name = (
            config.config_knowledge_manager.config_retrieval_mode.collection_name
        )
        use_output_from_verif_as_content = (
            config.config_content_scraper.use_output_from_verif_as_content
        )

        collection_name_formatted = f"{collection_name}--{chunk_size}--{embedder_model.name}---output_verif_as_content-{use_output_from_verif_as_content}"

        neural_searcher = NeuralSearcher(
            embedder_model=embedder_model,
            collection_name=collection_name_formatted,
        )

        dataset_langchain_docs = TextPreProcessor.create_chunk_dataset(
            content=text_sample_knowledge_content,
            src=src_sample_knowledge_content,
            chunk_size=chunk_size,
        )

        ids, payloads, texts = [], [], []
        for doc in dataset_langchain_docs:
            if len(doc.page_content) == 0:
                logger.warning(f"[process_record_content] len(doc.page_content) == 0")
                continue

            # check if vector already exists (vector close condition)
            # Tuple[List[qd_types.Record], Optional[qd_types.PointId]]
            already_exists, _ = neural_searcher.search_by_payload_scroll_must_values(
                must_value={
                    "src": src_sample_knowledge_content,
                    "text": doc.page_content,
                }
            )
            logger.info(
                f"already_exists = neural_searcher.search_by_payload_scroll_must_values: \n"
                f"LEN: {len(already_exists)}\n"
            )
            if len(already_exists) > 0:
                logger.warning(
                    f"already_exists = neural_searcher.search_by_payload_scroll_must_values: \n"
                    f"already_exists: {already_exists}; \n\n"
                )
                logger.warning(
                    f"already_exists = neural_searcher.search_by_payload_scroll_must_values: \n"
                    f"LEN: {len(already_exists)}\n"
                )
                continue

            texts.append(doc.page_content)
            ids.append(str(uuid.uuid4()))
            payloads.append({"src": src_sample_knowledge_content})

        if (len(ids) != 0) and (len(payloads) != 0) and (len(texts) != 0):

            neural_searcher.upload(
                ids=ids,
                payloads=payloads,
                texts=texts,
            )
        else:
            logger.warning(
                f"[process_record_content] "
                f"len(ids): {len(ids)}; "
                f"len(payloads): {len(payloads)}; "
                f"len(texts): {len(texts)}"
            )

        return {
            "status": True,
            # "error": None
        }
    except Exception as e:
        logger.exception(f"{e}")
        return {
            "status": False,
            "error": str(e),
        }
