"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict


class SnapshotsConfigInput(TypedDict):
    """Configuration section for snapshots."""

    graphml: NotRequired[bool | str | None]
    raw_entities: NotRequired[bool | str | None]
    top_level_nodes: NotRequired[bool | str | None]
