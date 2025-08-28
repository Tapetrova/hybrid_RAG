import json
import os
import traceback
from typing import Dict

import boto3
import pandas as pd
from io import StringIO
from apps.knowledge_manager.src.graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from apps.knowledge_manager.src.graphrag.index.progress.rich import RichProgressReporter
from libs.python.utils.logger import logger
from retry import retry
from botocore.exceptions import ClientError


def _get_progress_reporter(reporter_type: str | None) -> ProgressReporter:
    if reporter_type is None or reporter_type == "rich":
        return RichProgressReporter("GraphRAG Indexer ")
    if reporter_type == "print":
        return PrintProgressReporter("GraphRAG Indexer ")
    if reporter_type == "none":
        return NullProgressReporter()

    msg = f"Invalid progress reporter type: {reporter_type}"
    raise ValueError(msg)


class S3Helper:
    def __init__(self, region_name: str = None):
        # Create a boto3 client
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=(
                os.getenv("AWS_S3_REGION") if region_name is None else region_name
            ),
        )
        self._client_err = self.s3_client.exceptions.ClientError

    @retry(exceptions=(ClientError,), tries=5, delay=3)
    def upload_dataframe_to_s3(self, df: pd.DataFrame, bucket_name: str, s3_path: str):
        """
        Uploads a pandas DataFrame as a CSV file to an S3 bucket.

        Parameters:
        df (pd.DataFrame): The DataFrame to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_path (str): The path within the S3 bucket where the CSV will be stored.
        aws_access_key_id (str, optional): AWS access key ID.
        aws_secret_access_key (str, optional): AWS secret access key.
        region_name (str, optional): AWS region name.

        Returns:
        None
        """

        # Convert DataFrame to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        # Verify the upload
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            logger.info(f"File '{s3_path}' already exist in bucket '{bucket_name}'.")
            return True, None
        except self.s3_client.exceptions.ClientError:
            logger.info(
                f"File '{s3_path}' does not exist in bucket '{bucket_name}'. Start put new file."
            )
            # Upload the CSV to S3
            self.s3_client.put_object(
                Bucket=bucket_name, Key=s3_path, Body=csv_buffer.getvalue()
            )

        # Verify the upload
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            logger.info(
                f"File '{s3_path}' uploaded successfully to bucket '{bucket_name}'."
            )
            return True, None
        except self.s3_client.exceptions.ClientError as e:
            exc = traceback.format_exc()
            exc = f"ERROR:{e}:{exc}"
            logger.info(
                f"File '{s3_path}' failed to upload to bucket '{bucket_name}'. Error: {exc}"
            )
            return False, exc

    @retry(exceptions=(ClientError,), tries=5, delay=3)
    def upload_json_to_s3(self, json_data: Dict, bucket_name: str, s3_path: str):

        # Verify the upload
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            logger.info(f"File '{s3_path}' already exist in bucket '{bucket_name}'.")
            return True, None
        except self.s3_client.exceptions.ClientError:
            logger.info(
                f"File '{s3_path}' does not exist in bucket '{bucket_name}'. Start put new file."
            )
            # Upload the CSV to S3
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_path,
                Body=json.dumps(json_data),
                ContentType="application/json",
            )

        # Verify the upload
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            logger.info(
                f"File '{s3_path}' uploaded successfully to bucket '{bucket_name}'."
            )
            return True, None
        except self.s3_client.exceptions.ClientError as e:
            exc = traceback.format_exc()
            exc = f"ERROR:{e}:{exc}"
            logger.info(
                f"File '{s3_path}' failed to upload to bucket '{bucket_name}'. Error: {exc}"
            )
            return False, exc

    @retry(exceptions=(ClientError,), tries=10, delay=2)
    def check_file_existence(self, bucket_name: str, s3_path: str):
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=s3_path)
            logger.info(f"File '{s3_path}' exists in the bucket '{bucket_name}'.")
            return True

        except self.s3_client.exceptions.ClientError as e:
            exc = traceback.format_exc()
            exc = f"ERROR:{e}:{exc}"
            logger.info(
                f"File '{s3_path}' does NOT exist in the bucket '{bucket_name}'."
            )
            return False

    @retry(exceptions=(ClientError,), tries=10, delay=2)
    def read_json(self, bucket_name: str, s3_path: str) -> Dict | None:
        try:
            data = self.s3_client.get_object(Bucket=bucket_name, Key=s3_path)
            logger.info(f"File '{s3_path}' exists in the bucket '{bucket_name}'.")
            # Read the content of the file
            data = data["Body"].read().decode("utf-8")

            # Load JSON content
            json_data = json.loads(data)
            return json_data

        except Exception as e:
            exc = traceback.format_exc()
            exc = f"ERROR:{e}:{exc}"
            logger.info(f"!!!Cannot read File '{s3_path}'!!! \nEXC: {exc}.")
            return None
