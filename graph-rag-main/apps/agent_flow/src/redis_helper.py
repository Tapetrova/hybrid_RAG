import os
from typing import Optional, List, Any
from dataclasses import dataclass

import aioredis

from libs.python.utils.logger import logger


@dataclass
class RedisConfig:
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    password: str = "redispassword"
    db: int = 2
    decode_responses: bool = True

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD_LLM", "redispassword"),
            db=int(os.getenv("REDIS_DB_PUB_SUB", 2)),
            decode_responses=True,
        )


class RedisClientManager:
    """Manages Redis client instances."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._client: Optional[aioredis.Redis] = None

    @property
    def client(self) -> aioredis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = aioredis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                db=self.config.db,
                decode_responses=self.config.decode_responses,
            )
        return self._client

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None


# Create default Redis clients
_redis_config = RedisConfig.from_env()
_agent_client_manager = RedisClientManager(_redis_config)
_eval_client_manager = RedisClientManager(_redis_config)

REDIS_CLIENT_AGENT = _agent_client_manager.client
REDIS_CLIENT_EVAL = _eval_client_manager.client


class RedisSubscriber:
    """Handles Redis pub/sub subscriptions."""

    def __init__(self, client: aioredis.Redis):
        self.client = client

    async def subscribe_and_collect(
        self,
        channel_name: str,
        collected_events: List[Any],
        stop_on_status: str = "SUCCESS",
    ) -> None:
        """Subscribe to a channel and collect events until stop condition."""
        pubsub = self.client.pubsub()

        try:
            logger.info(f"Subscribing to channel: {channel_name}")
            await pubsub.subscribe(channel_name)

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                event_data = message.get("data")
                if event_data:
                    # Safely log event data
                    log_data = (
                        str(event_data)[:100]
                        if isinstance(event_data, (str, bytes))
                        else str(event_data)[:100]
                    )
                    logger.debug(f"Received message on {channel_name}: {log_data}...")

                    collected_events.append(event_data)

                    # Check stop condition
                    try:
                        if isinstance(event_data, str):
                            import json

                            event_dict = json.loads(event_data)
                            if event_dict.get("status") == stop_on_status:
                                logger.info(
                                    f"Stop condition met: status={stop_on_status}"
                                )
                                break
                    except (json.JSONDecodeError, AttributeError, KeyError) as e:
                        logger.debug(f"Failed to parse event data: {e}")
                        pass

        except Exception as e:
            logger.error(f"Error in subscribe_and_collect: {e}")
            raise

        finally:
            logger.info(f"Unsubscribing from channel: {channel_name}")
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")


# Keep the original function for backward compatibility
async def redis_subscriber(channel_name: str, collected_events: list):
    """Backward compatible wrapper for RedisSubscriber."""
    subscriber = RedisSubscriber(REDIS_CLIENT_AGENT)
    await subscriber.subscribe_and_collect(channel_name, collected_events)
