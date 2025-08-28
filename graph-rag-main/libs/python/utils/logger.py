# import os
# import time
# from pathlib import Path
import json

# import os
# import uuid

# import boto3
# import watchtower
import logging
import traceback
from copy import deepcopy


class TruncatedJSONFormatter(logging.Formatter):
    def __init__(
        self, max_list_length=3, max_dict_items=3, max_str_length=50, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.max_list_length = max_list_length
        self.max_dict_items = max_dict_items
        self.max_str_length = max_str_length

    def format(self, record):
        if isinstance(record.msg, (dict, list)):
            record.msg = self._truncate_json(record.msg)
            try:
                record.msg = json.dumps(record.msg, indent=2)
            except Exception as e:
                exc = traceback.format_exc()
                print(f"{e}:::{exc}")

        elif isinstance(record.msg, str):
            record.msg = self._truncate_string(record.msg)
        return super().format(record)

    def _truncate_json(self, obj):
        obj_ = deepcopy(obj)
        """Truncate lists, dicts, and strings in a JSON-like object"""
        if isinstance(obj_, dict):
            truncated_dict = {
                k: (
                    self._truncate_json(v)
                    if i < self.max_dict_items
                    else "... and_so_on"
                )
                for i, (k, v) in enumerate(obj_.items())
                if i < self.max_dict_items
            }
            return truncated_dict
        elif isinstance(obj_, list):
            truncated_list = [
                self._truncate_json(item)
                for i, item in enumerate(obj_)
                if i < self.max_list_length
            ]
            if len(obj_) > self.max_list_length:
                truncated_list.append("... and_so_on")
            return truncated_list
        elif isinstance(obj_, str):
            return self._truncate_string(obj_)
        return obj_

    def _truncate_string(self, string):
        """Truncate a string if it's too long"""
        if len(string) > self.max_str_length:
            return string[: self.max_str_length] + "... and_so_on"
        return string


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# gino off logs
logging.getLogger("gino.engine._SAEngine").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Set up the handler and formatter
handler = logging.StreamHandler()

# Use the TruncatedJSONFormatter for structured JSON output
formatter = TruncatedJSONFormatter(
    max_list_length=8, max_dict_items=8, max_str_length=250
)
handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(handler)

# formatter = logging.Formatter(
#     fmt="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# s3_handler = watchtower.CloudWatchLogHandler(
#     boto3_client=boto3.client(
#         service_name="logs",
#         aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#         aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         region_name=os.getenv("AWS_S3_REGION"),
#     ),
#     log_group=os.getenv("LOG_GROUP", "logs"),
#     stream_name=str(uuid.uuid4()),
#     create_log_group=True,
#     create_log_stream=True,
# )
# s3_handler.setFormatter(formatter)
# logger.addHandler(s3_handler)

# logging_file = (
#     Path(os.getcwd()) / "reports" / f"logs_{time.strftime('%Y%m%d-%H%M%S')}.log"
# )
# logging_file.parent.mkdir(parents=True, exist_ok=True)
#
# logging_file.touch(exist_ok=True)
