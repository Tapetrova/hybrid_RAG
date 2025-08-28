import hashlib
import json
import os
import redis
from libs.python.utils.logger import logger

CACHE_EXPIRATION = os.getenv("CACHE_EXPIRATION", 3600)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_DB_CONV = os.getenv("REDIS_DB_CONV", 7)
REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_PASSWORD_LLM = os.getenv("REDIS_PASSWORD_LLM", "redispassword")

redis_password_filled = "Yes" if len(REDIS_PASSWORD_LLM) > 0 else "No"

logger.info(f"{REDIS_HOST}, {REDIS_PORT}, {REDIS_DB_CONV}, {redis_password_filled}")

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_CONV, password=REDIS_PASSWORD_LLM
)
redis_client_async = redis.asyncio.Redis(
    host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_CONV, password=REDIS_PASSWORD_LLM
)


def hash_string(
    input_string,
    algorithm="md5",
):
    """
    Hashes an input string using the specified algorithm.

    :param input_string: The string to be hashed.
    :param algorithm: The hashing algorithm to use (default is 'sha256').
    :return: The hexadecimal hash of the input string.
    """
    # Create a new hash object using the specified algorithm
    hash_object = hashlib.new(algorithm)

    # Update the hash object with the bytes of the input string
    hash_object.update(input_string.encode())

    # Return the hexadecimal representation of the hash
    return hash_object.hexdigest()


def cache_data(cache_name, data, ex=CACHE_EXPIRATION):
    redis_client.set(cache_name, json.dumps(data), ex=ex)
    return True


def get_cached_data(cache_name):
    cached_data = redis_client.get(cache_name)
    return json.loads(cached_data) if cached_data is not None else None


async def acache_data(cache_name, data, ex=CACHE_EXPIRATION):
    await redis_client_async.set(cache_name, json.dumps(data), ex=ex)
    return True


async def aget_cached_data(cache_name):
    cached_data = await redis_client_async.get(cache_name)
    return json.loads(cached_data) if cached_data is not None else None
