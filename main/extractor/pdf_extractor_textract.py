import boto3
import logging
import time
from collections import defaultdict
from main.config import Config
from .pdf_extractor_base import PDFExtractorBase


logger = logging.getLogger(__name__)


class TextractExtractor(PDFExtractorBase):
    """Extract text from PDFs using AWS Textract."""

    def __init__(self, region="us-east-1", poll_interval=5, timeout=300):
        self.client = boto3.client("textract", region_name=region)
        self.poll_interval = poll_interval
        self.timeout = timeout


    def extract_text(self, file_path_or_key: str) -> str:
        try:
            if Config.USE_S3:
                # --- Async mode for S3 with TABLES and FORMS ---
                response = self.client.start_document_analysis(
                    DocumentLocation={
                        "S3Object": {
                            "Bucket": Config.S3_BUCKET,
                            "Name": file_path_or_key,
                        }
                    },
                    FeatureTypes=["TABLES", "FORMS"]
                )
                job_id = response["JobId"]

                # Poll for job completion
                start_time = time.time()
                while True:
                    result = self.client.get_document_analysis(JobId=job_id)
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
                    page = self.client.get_document_analysis(**args)
                    blocks.extend(page.get("Blocks", []))
                    next_token = page.get("NextToken")
                    if not next_token:
                        break

            else:
                # --- Sync mode for local files ---
                with open(file_path_or_key, "rb") as f:
                    response = self.client.analyze_document(
                        Document={"Bytes": f.read()},
                        FeatureTypes=["TABLES", "FORMS"]
                    )

                blocks = response.get("Blocks", [])

            # Extract both LINE and CELL text
            lines = [item["Text"] for item in blocks if item["BlockType"] == "LINE" and "Text" in item]
            cells = [item["Text"] for item in blocks if item["BlockType"] == "CELL" and "Text" in item]
            reconstructed_lines = self.reconstruct_lines_from_words(blocks)
            text = "\n".join(lines + cells + reconstructed_lines)

            return text

        except Exception as e:
            raise RuntimeError(f"Textract failed to extract text: {e}") from e



    def reconstruct_lines_from_words(self, blocks):
        """Reconstruct lines from WORD blocks using bounding box proximity."""
        lines_by_page = defaultdict(list)

        for block in blocks:
            if block["BlockType"] == "WORD" and "Text" in block and "Geometry" in block:
                page = block.get("Page", 1)
                top = block["Geometry"]["BoundingBox"]["Top"]
                left = block["Geometry"]["BoundingBox"]["Left"]
                lines_by_page[page].append((top, left, block["Text"]))

        reconstructed = []
        for page, words in lines_by_page.items():
            # Sort by vertical position (Top), then horizontal (Left)
            words.sort()
            current_line = []
            last_top = None

            for top, left, text in words:
                if last_top is None or abs(top - last_top) < 0.005:  # same line
                    current_line.append((left, text))
                else:
                    # New line
                    current_line.sort()
                    reconstructed.append(" ".join(t for _, t in current_line))
                    current_line = [(left, text)]
                last_top = top

            if current_line:
                current_line.sort()
                reconstructed.append(" ".join(t for _, t in current_line))

        return reconstructed
