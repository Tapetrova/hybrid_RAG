from __future__ import annotations

from typing import Any

from libs.python.utils.logger import logger
from .pipeline_cache import PipelineCache

# from libs.python.utils.cacher import aget_cached_data, acache_data, redis_client_async
from libs.python.utils.cacher import get_cached_data, cache_data, redis_client


class RedisPipelineCache(PipelineCache):
    """Redis cache class definition."""

    _name: str

    def __init__(self, name: str | None = None):
        """Init method definition."""
        self._name = name or "cache"

    async def get(self, key: str) -> Any:
        """Get the value for the given key.

        Args:
            - key - The key to get the value for.

        Returns
        -------
            - output - The value for the given key.
        """
        key = self._create_cache_key(key)
        # data = await aget_cached_data(key)
        data = get_cached_data(key)
        logger.info(f"GET cache data: {key}: data_type: {type(data)};")
        if data is None:
            return data
        else:
            return data.get("result")

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set the value for the given key.

        Args:
            - key - The key to set the value for.
            - value - The value to set.
        """
        key = self._create_cache_key(key)

        if value is None:
            return

        data = {"result": value, **(debug_data or {})}

        # await acache_data(key, data, ex=1280000)
        cache_data(key, data, ex=15360000)  # 177.777 days
        logger.info(f"SET cache data: {key}: data_type: {type(data)};")

    async def has(self, key: str) -> bool:
        """Return True if the given key exists in the cache.

        Args:
            - key - The key to check for.

        Returns
        -------
            - output - True if the key exists in the cache, False otherwise.
        """
        key = self._create_cache_key(key)
        # return await redis_client_async.exists(key) > 0
        logger.info(
            f"CHECK HAS cache data: {key}: RESULT: {redis_client.exists(key) > 0};"
        )
        return redis_client.exists(key) > 0

    async def delete(self, key: str) -> None:
        """Delete the given key from the cache.

        Args:
            - key - The key to delete.
        """
        key = self._create_cache_key(key)
        # await redis_client_async.delete(key)
        redis_client.delete(key)
        logger.info(f"DELETE cache data {key};")

    async def clear(self) -> None:
        """Clear the cache."""
        # keys = await redis_client_async.keys(f"{self._name}*")
        keys = redis_client.keys(f"{self._name}*")
        if keys:
            # await redis_client_async.delete(*keys)
            redis_client.delete(*keys)
        logger.info(f"CLEAR ALL cache data len_keys: {len(keys)};")

    def child(self, name: str) -> PipelineCache:
        """Create a child cache with the given name.

        Args:
            - name - The name to create the sub cache with.
        """
        logger.info(f"GET CHILD cache data name: {name};")
        return RedisPipelineCache(f"{self._name}{name}")

    def _create_cache_key(self, key: str) -> str:
        """Create a cache key for the given key."""
        return f"{self._name}{key}"
