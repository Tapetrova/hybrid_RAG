"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict

from libs.python.schemas.enums import StorageType


class StorageConfigInput(TypedDict):
    """The default configuration section for Storage."""

    type: NotRequired[StorageType | str | None]
    base_dir: NotRequired[str | None]
    connection_string: NotRequired[str | None]
    container_name: NotRequired[str | None]
    storage_account_blob_url: NotRequired[str | None]
    bucket_name: NotRequired[str | None]
    region_name: NotRequired[str | None]
    object_name: NotRequired[str | None]
