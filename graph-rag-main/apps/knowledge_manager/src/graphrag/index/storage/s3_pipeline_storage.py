"""AWS S3 implementation of PipelineStorage."""

import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from apps.knowledge_manager.src.graphrag.index.progress import ProgressReporter
from datashaper import Progress
from .typing import PipelineStorage

from libs.python.utils.logger import logger

log = logger


class S3PipelineStorage(PipelineStorage):
    """The S3-Storage implementation."""

    def __init__(
        self,
        bucket_name: str,
        region_name: str | None = None,
        encoding: str | None = None,
        path_prefix: str | None = None,
    ):
        """Create a new S3Storage instance."""
        self._bucket_name = bucket_name
        self._region_name = region_name
        self._encoding = encoding or "utf-8"
        self._path_prefix = path_prefix or ""

        self._s3_client = boto3.client(
            "s3",
            region_name=self._region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        log.info(
            "creating S3 storage at bucket=%s, path=%s",
            self._bucket_name,
            self._path_prefix,
        )
        self.create_bucket()

    def create_bucket(self) -> None:
        """Create the bucket if it does not exist."""
        if not self.bucket_exists():
            try:
                self._s3_client.create_bucket(
                    Bucket=self._bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": self._region_name},
                )
            except ClientError as e:
                if e.response["Error"]["Code"] == "BucketAlreadyExists":
                    log.warning(f"Bucket {self._bucket_name} already exists.")
                else:
                    log.exception("Error creating bucket: %s", e)
                    raise

    def bucket_exists(self) -> bool:
        """Check if the bucket exists."""
        try:
            self._s3_client.head_bucket(Bucket=self._bucket_name)
            return True
        except ClientError:
            return False

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        progress: ProgressReporter | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find files in the storage using a file pattern, as well as a custom filter function."""
        base_dir = base_dir or ""

        log.info(
            "search bucket %s for files matching %s",
            self._bucket_name,
            file_pattern.pattern,
        )

        def blobname(blob_name: str) -> str:
            if blob_name.startswith(self._path_prefix):
                blob_name = blob_name.replace(self._path_prefix, "", 1)
            if blob_name.startswith("/"):
                blob_name = blob_name[1:]
            return blob_name

        def item_filter(key_name: str, file_filter_: dict[str, Any]) -> bool:
            if file_filter_ is None:
                return True

            conditions = []
            for name, flag in file_filter_.items():
                if flag:
                    if name in key_name:
                        conditions.append(True)
                    else:
                        conditions.append(False)
                else:
                    if name not in key_name:
                        conditions.append(True)
                    else:
                        conditions.append(False)

            return all(conditions)

        try:
            paginator = self._s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self._bucket_name, Prefix=base_dir
            )

            num_loaded = 0
            num_total = 0
            num_filtered = 0

            for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        num_total += 1
                        match = file_pattern.match(obj["Key"])
                        if match and obj["Key"].startswith(base_dir):
                            if item_filter(
                                key_name=obj["Key"], file_filter_=file_filter
                            ):
                                yield (blobname(obj["Key"]), dict())
                                num_loaded += 1
                                if max_count > 0 and num_loaded >= max_count:
                                    return
                            else:
                                num_filtered += 1
                        else:
                            num_filtered += 1
                        if progress is not None:
                            progress(
                                _create_progress_status(
                                    num_loaded, num_filtered, num_total
                                )
                            )
        except Exception:
            log.exception(
                "Error finding objects: base_dir=%s, file_pattern=%s, file_filter=%s",
                base_dir,
                file_pattern,
                file_filter,
            )
            raise

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get a value from the storage."""
        try:
            key = self._keyname(key)
            response = self._s3_client.get_object(Bucket=self._bucket_name, Key=key)
            data = response["Body"].read()
            if not as_bytes:
                coding = encoding or self._encoding
                data = data.decode(coding)
        except Exception:
            log.exception("Error getting key %s", key)
            return None
        else:
            return data

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set a value in the storage."""
        try:
            key = self._keyname(key)
            if isinstance(value, bytes):
                self._s3_client.put_object(
                    Bucket=self._bucket_name, Key=key, Body=value
                )
            else:
                coding = encoding or self._encoding
                self._s3_client.put_object(
                    Bucket=self._bucket_name, Key=key, Body=value.encode(coding)
                )
        except Exception:
            log.exception(f"Error setting key {key}")

    async def has(self, key: str) -> bool:
        """Check if a key exists in the storage."""
        key = self._keyname(key)
        try:
            self._s3_client.head_object(Bucket=self._bucket_name, Key=key)
            return True
        except ClientError:
            return False

    async def delete(self, key: str) -> None:
        """Delete a key from the storage."""
        key = self._keyname(key)
        try:
            self._s3_client.delete_object(Bucket=self._bucket_name, Key=key)
        except ClientError as e:
            log.exception("Error deleting key %s: %s", key, e)

    async def clear(self) -> None:
        """Clear the storage."""
        try:
            objects_to_delete = self._s3_client.list_objects_v2(
                Bucket=self._bucket_name
            )
            if "Contents" in objects_to_delete:
                for obj in objects_to_delete["Contents"]:
                    self._s3_client.delete_object(
                        Bucket=self._bucket_name, Key=obj["Key"]
                    )
        except ClientError as e:
            log.exception("Error clearing bucket: %s", e)

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        if name is None:
            return self
        path = str(Path(self._path_prefix) / name)
        return S3PipelineStorage(
            self._bucket_name,
            self._region_name,
            self._encoding,
            path,
        )

    def _keyname(self, key: str) -> str:
        """Get the key name."""
        return str(Path(self._path_prefix) / key)


def create_s3_storage(
    bucket_name: str,
    region_name: str | None = None,
    base_dir: str | None = None,
) -> PipelineStorage:
    """Create an S3 based storage."""
    log.info("Creating S3 storage at %s", bucket_name)
    if bucket_name is None:
        msg = "No bucket name provided for S3 storage."
        raise ValueError(msg)
    return S3PipelineStorage(
        bucket_name,
        region_name=region_name,
        path_prefix=base_dir,
    )


def validate_s3_bucket_name(bucket_name: str) -> bool:
    """
    Check if the provided S3 bucket name is valid based on AWS rules.

        - An S3 bucket name must be between 3 and 63 characters in length.
        - Start with a lowercase letter or number.
        - Can contain only lowercase letters, numbers, dots (.), and hyphens (-).
        - Cannot be formatted as an IP address (e.g., 192.168.1.1).
        - Cannot contain uppercase characters or underscores.

    Args:
    -----
    bucket_name (str)
        The S3 bucket name to be validated.

    Returns
    -------
        bool: True if valid, False otherwise.
    """
    # Check the length of the name
    if len(bucket_name) < 3 or len(bucket_name) > 63:
        raise ValueError(
            f"Bucket name must be between 3 and 63 characters in length. Name provided was {len(bucket_name)} characters long."
        )

    # Check if the name starts with a lowercase letter or number
    if not bucket_name[0].islower() and not bucket_name[0].isdigit():
        raise ValueError(
            f"Bucket name must start with a lowercase letter or number. Starting character was {bucket_name[0]}."
        )

    # Check for valid characters (lowercase letters, numbers, dots, hyphens)
    if not re.match("^[a-z0-9.-]+$", bucket_name):
        raise ValueError(
            f"Bucket name must only contain:\n- lowercase letters\n- numbers\n- dots\n- hyphens\nName provided was {bucket_name}."
        )

    # Check if the name is formatted as an IP address
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", bucket_name):
        raise ValueError(
            f"Bucket name cannot be formatted as an IP address. Name provided was {bucket_name}."
        )

    return True


def _create_progress_status(
    num_loaded: int, num_filtered: int, num_total: int
) -> Progress:
    return Progress(
        total_items=num_total,
        completed_items=num_loaded + num_filtered,
        description=f"{num_loaded} files loaded ({num_filtered} filtered)",
    )
