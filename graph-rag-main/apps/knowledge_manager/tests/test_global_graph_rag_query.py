# import cProfile
# import pstats
# import os
# import sys

import time


def run():
    import os
    from dotenv import load_dotenv

    st = time.time()
    from apps.knowledge_manager.src.graph_rag_manager.graph_rag_query import (
        GlobalSearchGraphRAGQueryManager,
    )

    print(f"[0]imported: {time.time() - st}")
    from libs.python.schemas.configuration import (
        Config,
        ConfigAgentFlow,
        ConfigKnowledgeManager,
        KnowledgeManagerMode,
    )
    from libs.python.schemas.graphrag import GraphRagConfig

    # load_dotenv(".env.local")
    load_dotenv(".env.local")

    st = time.time()
    cfg = Config(
        config_agent_flow=ConfigAgentFlow(),
        config_knowledge_manager=ConfigKnowledgeManager(
            mode=KnowledgeManagerMode.GRAPHRAG, config_retrieval_mode=GraphRagConfig()
        ),
        config_content_scraper=None,
    )
    print(f"[0]inited cfg: {time.time() - st}")

    st = time.time()
    search_engine = GlobalSearchGraphRAGQueryManager(config=cfg)
    print(f"[0]inited GlobalSearchGraphRAGQueryManager: {time.time() - st}")

    st = time.time()
    result = search_engine.search_global(
        # "What's the difference between Subaru Outback and Volkswagen Touareg in terms of engine?"
        "Subaru Outback 2023 key features"
    )
    print(f"[0]result = search_engine.search: {time.time() - st}")

    print(result.response)
    print(result.completion_time)


if __name__ == "__main__":
    stt = time.time()
    run()
    print(f"total: {time.time() - stt}")
    # with cProfile.Profile() as profile:
    #     run()
    #
    # profile_result = pstats.Stats(profile)
    # profile_result.print_stats(pstats.SortKey.TIME)
