"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict

from libs.python.schemas.enums import CacheType


class CacheConfigInput(TypedDict):
    """The default configuration section for Cache."""

    type: NotRequired[CacheType | str | None]
    base_dir: NotRequired[str | None]
    connection_string: NotRequired[str | None]
    container_name: NotRequired[str | None]
    storage_account_blob_url: NotRequired[str | None]
    bucket_name: NotRequired[str | None]
    region_name: NotRequired[str | None]
