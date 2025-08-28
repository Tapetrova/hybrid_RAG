import hashlib

from typing import (
    Any,
    Optional,
)

from libs.python.utils.logger import logger

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain.load.dump import dumps
from langchain.load.load import loads
from langchain.schema import Generation
from langchain.schema.cache import RETURN_VAL_TYPE, BaseCache


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.md5(_input.encode()).hexdigest()


class RedisCache(BaseCache):
    """Cache that uses Redis as a backend."""

    def __init__(self, redis_: Any, *, ttl: Optional[int] = None):
        """
        Initialize an instance of RedisCache..

        This method initializes an object with Redis caching capabilities.
        It takes a `redis_` parameter, which should be an instance of a Redis
        client class, allowing the object to interact with a Redis
        server for caching purposes.

        Parameters:
            redis_ (Any): An instance of a Redis client class
                (e.g., redis.Redis) used for caching.
                This allows the object to communicate with a
                Redis server for caching operations.
            ttl (int, optional): Time-to-live (TTL) for cached items in seconds.
                If provided, it sets the time duration for how long cached
                items will remain valid. If not provided, cached items will not
                have an automatic expiration.
        """
        try:
            from redis import Redis
        except ImportError:
            raise ValueError(
                "Could not import redis python package. "
                "Please install it with `pip install redis`."
            )
        if not isinstance(redis_, Redis):
            raise ValueError("Please pass in Redis object.")
        self.redis = redis_
        self.ttl = ttl

    def _key(self, prompt: str, llm_string: str) -> str:
        """Compute key from prompt and llm_string"""
        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        generations = []
        # Read from a Redis HASH
        results = self.redis.hgetall(self._key(prompt, llm_string))
        if results:
            for _, text in results.items():
                try:
                    generations.append(loads(text))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.append(Generation(text=text))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "RedisCache only supports caching of normal LLM generations, "
                    f"got {type(gen)}"
                )
        # Write to a Redis HASH
        key = self._key(prompt, llm_string)

        with self.redis.pipeline() as pipe:
            pipe.hset(
                key,
                mapping={
                    str(idx): dumps(generation)
                    for idx, generation in enumerate(return_val)
                },
            )
            if self.ttl is not None:
                pipe.expire(key, self.ttl)

            pipe.execute()

    def clear(self, **kwargs: Any) -> None:
        """Clear cache. If `asynchronous` is True, flush asynchronously."""
        asynchronous = kwargs.get("asynchronous", False)
        self.redis.flushdb(asynchronous=asynchronous, **kwargs)
