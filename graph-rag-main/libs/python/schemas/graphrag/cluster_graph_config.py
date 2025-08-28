"""Parameterization settings for the default configuration."""

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs


class ClusterGraphConfig(BaseModelUpd):
    """Configuration section for clustering graphs."""

    max_cluster_size: int = Field(
        description="The maximum cluster size to use.", default=defs.MAX_CLUSTER_SIZE
    )
    strategy: dict | None = Field(
        description="The cluster strategy to use.", default=None
    )

    def resolved_strategy(self) -> dict:
        """Get the resolved cluster strategy."""
        from apps.knowledge_manager.src.graphrag.index.verbs.graph.clustering import (
            GraphCommunityStrategyType,
        )

        return self.strategy or {
            "type": GraphCommunityStrategyType.leiden,
            "max_cluster_size": self.max_cluster_size,
        }
