"""A reporter that writes to S3 storage."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError
from datashaper import NoopWorkflowCallbacks


class S3WorkflowCallbacks(NoopWorkflowCallbacks):
    """A reporter that writes to S3 storage."""

    _max_block_count: int = 25_000  # 25k blocks per object

    def __init__(
        self,
        bucket_name: str,
        region_name: str | None = None,
        object_name: str = "",
        base_dir: str | None = None,
    ):  # type: ignore
        """Create a new instance of the S3StorageReporter class."""
        if bucket_name is None:
            msg = "No bucket name provided for S3 storage."
            raise ValueError(msg)

        self._bucket_name = bucket_name
        self._region_name = region_name

        self._s3_client = boto3.client(
            "s3",
            region_name=self._region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        if (object_name == "") or (object_name is None):
            object_name = f"report/{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H:%M:%S:%f')}.logs.json"

        self._object_name = str(Path(base_dir or "") / object_name)

        self._num_blocks = 0  # refresh block counter
        self._logs = []

    def _write_log(self, log: dict[str, Any]):
        # create a new file when block count hits close to 25k
        if self._num_blocks >= self._max_block_count:
            self._flush_logs()
            new_base_dir = str(Path(self._object_name).parent)
            self.__init__(self._bucket_name, self._region_name, base_dir=new_base_dir)

        self._logs.append(json.dumps(log) + "\n")
        self._num_blocks += 1

        if self._num_blocks >= self._max_block_count:
            self._flush_logs()

    def _flush_logs(self):
        if self._logs:
            try:
                self._s3_client.put_object(
                    Bucket=self._bucket_name,
                    Key=self._object_name,
                    Body="".join(self._logs),
                )
                self._logs = []
            except ClientError as e:
                print(f"Failed to write logs to S3: {e}")
                raise

    def on_error(
        self,
        message: str,
        cause: BaseException | None = None,
        stack: str | None = None,
        details: dict | None = None,
    ):
        """Report an error."""
        self._write_log(
            {
                "type": "error",
                "data": message,
                "cause": str(cause),
                "stack": stack,
                "details": details,
            }
        )

    def on_warning(self, message: str, details: dict | None = None):
        """Report a warning."""
        self._write_log({"type": "warning", "data": message, "details": details})

    def on_log(self, message: str, details: dict | None = None):
        """Report a generic log message."""
        self._write_log({"type": "log", "data": message, "details": details})
