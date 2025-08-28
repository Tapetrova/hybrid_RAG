import os
from urllib.parse import quote

from celery import Celery

redis_password_encoded = quote(os.getenv("REDIS_PASSWORD_LLM"))

CELERY_BROKER = f"redis://default:{redis_password_encoded}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('CELERY_BROKER_KNOWLEDGE_MANAGER_DB')}"
CELERY_BACKEND = f"redis://default:{redis_password_encoded}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('CELERY_BACKEND_KNOWLEDGE_MANAGER_DB')}"


celery_app = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)
