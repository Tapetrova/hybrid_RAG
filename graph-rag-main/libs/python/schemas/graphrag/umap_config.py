"""Parameterization settings for the default configuration."""

from libs.python.schemas.basic_models import BaseModelUpd, BaseEnum, Field

import libs.python.schemas.defaults as defs


class UmapConfig(BaseModelUpd):
    """Configuration section for UMAP."""

    enabled: bool = Field(
        description="A flag indicating whether to enable UMAP.",
        default=defs.UMAP_ENABLED,
    )
