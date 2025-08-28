"""Common default configuration values."""

import os

from libs.python.schemas.enums import (
    AsyncType,
    CacheType,
    InputFileType,
    InputType,
    LLMType,
    ReportingType,
    StorageType,
    TextEmbeddingTarget,
)
from libs.python.schemas.basic_models import LLMModel

BUCKET_NAME = "dev-1"
AWS_S3_REGION = os.getenv("AWS_S3_REGION", "eu-north-1")

ASYNC_MODE = AsyncType.Threaded
ENCODING_MODEL = "cl100k_base"
#
# LLM Parameters
#
LLM_TYPE = LLMType.OpenAIChat
LLM_MODEL = LLMModel.GPT_4O_MINI
LLM_MAX_TOKENS = 4000
LLM_TEMPERATURE = 0
LLM_TOP_P = 1
LLM_N = 1
LLM_REQUEST_TIMEOUT = 180.0
LLM_TOKENS_PER_MINUTE = 0
LLM_REQUESTS_PER_MINUTE = 0
LLM_MAX_RETRIES = 10
LLM_MAX_RETRY_WAIT = 10.0
LLM_SLEEP_ON_RATE_LIMIT_RECOMMENDATION = True
LLM_CONCURRENT_REQUESTS = 25

#
# Text Embedding Parameters
#
EMBEDDING_TYPE = LLMType.OpenAIEmbedding
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_BATCH_MAX_TOKENS = 8191
EMBEDDING_TARGET = TextEmbeddingTarget.required

CACHE_TYPE = CacheType.redis
CACHE_BASE_DIR = "cache_{hash_data}"
CACHE_BASE_NAME = ""
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100
CHUNK_GROUP_BY_COLUMNS = ["id"]
CLAIM_DESCRIPTION = (
    "Any claims or facts that could be relevant to information discovery."
)
CLAIM_MAX_GLEANINGS = 0
CLAIM_EXTRACTION_ENABLED = True
MAX_CLUSTER_SIZE = 10
COMMUNITY_REPORT_MAX_LENGTH = 2000
COMMUNITY_REPORT_MAX_INPUT_LENGTH = 100_000
ENTITY_EXTRACTION_ENTITY_TYPES = [
    "Accidents",
    "Author",
    "Body",
    "Car",
    "Competitors",
    "Costs",
    "Drivetrain",
    "Engine",
    "Exterior",
    "Features",
    "Generation",
    "Interior",
    "Make",
    "Mileage",
    "Model",
    "Opinion",
    "Owners",
    "Package",
    "Performance",
    "Predecessor",
    "Segment",
    "Series",
    "Size",
    "Source",
    "Successor",
    "Transmission",
    "User",
    "Weight",
    "Year",
]
ENTITY_EXTRACTION_MAX_GLEANINGS = 0
INPUT_FILE_TYPE = InputFileType.csv
INPUT_TYPE = InputType.s3
INPUT_BASE_DIR = "input"
INPUT_FILE_ENCODING = "utf-8"
INPUT_TEXT_COLUMN = "content"
INPUT_CSV_PATTERN = ".*\\.csv$"
INPUT_TEXT_PATTERN = ".*\\.txt$"
PARALLELIZATION_STAGGER = 0.3
PARALLELIZATION_NUM_THREADS = 50
NODE2VEC_ENABLED = False
NODE2VEC_NUM_WALKS = 10
NODE2VEC_WALK_LENGTH = 40
NODE2VEC_WINDOW_SIZE = 2
NODE2VEC_ITERATIONS = 3
NODE2VEC_RANDOM_SEED = 597832
REPORTING_TYPE = ReportingType.s3
REPORTING_BASE_DIR = "output/{timestamp}/reports"
SNAPSHOTS_GRAPHML = True
SNAPSHOTS_RAW_ENTITIES = True
SNAPSHOTS_TOP_LEVEL_NODES = True
STORAGE_BASE_DIR = "output/{timestamp}/artifacts"
STORAGE_TYPE = StorageType.s3
SUMMARIZE_DESCRIPTIONS_MAX_LENGTH = 500
UMAP_ENABLED = False

# Local Search
LOCAL_SEARCH_TEXT_UNIT_PROP = 0.5 - 0.5
LOCAL_SEARCH_COMMUNITY_PROP = 0.1 + 0.5
LOCAL_SEARCH_CONVERSATION_HISTORY_MAX_TURNS = 5
LOCAL_SEARCH_TOP_K_MAPPED_ENTITIES = 10
LOCAL_SEARCH_TOP_K_RELATIONSHIPS = 10
LOCAL_SEARCH_MAX_TOKENS = 12_000
LOCAL_SEARCH_LLM_MAX_TOKENS = 2000

# Global Search
GLOBAL_SEARCH_MAX_TOKENS = 12_000
GLOBAL_SEARCH_DATA_MAX_TOKENS = 12_000
GLOBAL_SEARCH_MAP_MAX_TOKENS = 1000
GLOBAL_SEARCH_REDUCE_MAX_TOKENS = 2_000
GLOBAL_SEARCH_CONCURRENCY = 32
