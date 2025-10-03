import boto3
import time
from main.config import Config
from .pdf_extractor_base import PDFExtractorBase


class TextractExtractor(PDFExtractorBase):
    """Extract text from PDFs using AWS Textract."""

    def __init__(self, region="us-east-1", poll_interval=5, timeout=300):
        self.client = boto3.client("textract", region_name=region)
        self.poll_interval = poll_interval
        self.timeout = timeout

    def extract_text(self, file_path_or_key: str) -> str:
        try:
            if Config.USE_S3:
                # --- Async mode for S3 ---
                response = self.client.start_document_text_detection(
                    DocumentLocation={
                        "S3Object": {
                            "Bucket": Config.S3_BUCKET,
                            "Name": file_path_or_key,  # pass S3 key
                        }
                    }
                )
                job_id = response["JobId"]

                # Poll for job completion
                start_time = time.time()
                while True:
                    result = self.client.get_document_text_detection(JobId=job_id)
                    status = result["JobStatus"]

                    if status == "SUCCEEDED":
                        break
                    elif status in ["FAILED", "PARTIAL_SUCCESS"]:
                        raise RuntimeError(f"Textract async job failed: {status}")

                    if time.time() - start_time > self.timeout:
                        raise TimeoutError("Textract job timed out")

                    time.sleep(self.poll_interval)

                # Collect all pages
                blocks = []
                next_token = None
                while True:
                    args = {"JobId": job_id}
                    if next_token:
                        args["NextToken"] = next_token
                    page = self.client.get_document_text_detection(**args)
                    blocks.extend(page.get("Blocks", []))
                    next_token = page.get("NextToken")
                    if not next_token:
                        break

                # Async uses "Text"
                text = "\n".join(
                    item["Text"] for item in blocks if item["BlockType"] == "LINE" and "Text" in item
                )

            else:
                # --- Sync mode for local files ---
                with open(file_path_or_key, "rb") as f:
                    response = self.client.detect_document_text(Document={"Bytes": f.read()})

                # Sync uses "DetectedText"
                text = "\n".join(
                    item["DetectedText"]
                    for item in response["Blocks"]
                    if item["BlockType"] == "LINE" and "DetectedText" in item
                )

            return text

        except Exception as e:
            raise RuntimeError(f"Textract failed to extract text: {e}") from e
