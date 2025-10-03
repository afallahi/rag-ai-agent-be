import boto3
import os
import tempfile
from main.config import Config

def list_pdfs_in_bucket(bucket: str = Config.S3_BUCKET, prefix: str = Config.S3_PREFIX) -> list[str]:
    """Return a list of PDF files in the given S3 bucket/prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(".pdf"):
                files.append(obj["Key"])
    return files


def download_pdf(s3_key: str, bucket: str = Config.S3_BUCKET) -> str:
    """Download a PDF from S3 to a temporary file and return the local path."""
    s3 = boto3.client("s3")
    _, ext = os.path.splitext(s3_key)
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    s3.download_file(bucket, s3_key, tmp_path)
    return tmp_path
