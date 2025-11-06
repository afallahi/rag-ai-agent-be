import boto3
import os
import hashlib
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


def download_pdf(s3_key: str, bucket: str = Config.S3_BUCKET, cache_mode: str = None) -> str:
    """
    Download a PDF from S3 to a structured local cache path and return the local path.
    If USE_S3 is False, return the local sample path instead.
    If cache_mode is 'ephemeral', mark the file for deletion after indexing.
    If cache_mode is 'none', raise NotImplementedError (streaming not supported yet).
    """
    cache_mode = cache_mode or Config.CACHE_MODE
    if not Config.USE_S3:
        return os.path.join(Config.SAMPLE_DIR, s3_key)

    if cache_mode == "none":
        raise NotImplementedError("cache_mode='none' is not yet supported")

    local_path = os.path.join(Config.CACHE_DIR, s3_key)

    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(bucket, s3_key, local_path)

    if cache_mode == "ephemeral":
        return local_path + "::ephemeral"

    return local_path


def download_pdf_stream(s3_key: str, bucket: str = Config.S3_BUCKET) -> bytes:
    """Download a PDF from S3 and return its raw bytes."""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=s3_key)
    return response["Body"].read()


def hash_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
