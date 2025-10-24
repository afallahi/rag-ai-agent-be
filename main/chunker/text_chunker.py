"""General-purpose Text Chunking Module with heading-list preservation."""
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from main.config import Config

logger = logging.getLogger(__name__)



def prepare_for_chunking(text: str) -> str:
    """Normalize raw text for consistent chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize patterns like "E7 2" → "E7.2"
    text = re.sub(r"\b(E)\s+(\d+\.\d+)\b", r"\1\2", text)

    # Ensure bullet-like items start on new lines
    text = re.sub(r"(?<!\n)[•\-–\u2022]\s*", r"\n\g<0>", text)

    # Ensure numbered lists (1., 2.) start on new lines
    text = re.sub(r"(?<!\n)(\d+\.)\s+", r"\n\1 ", text)

    # Collapse redundant whitespace/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> list[str]:
    """
    Splits text into chunks while preserving heading + list or table proximity.
    This works for any document structure, not just specific sections.
    """
    text = prepare_for_chunking(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n•", "\n-", "\n", " ", ""],
    )
    chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

    merged_chunks = merge_heading_with_following_list(chunks)
    return merged_chunks


def merge_heading_with_following_list(chunks: list[str]) -> list[str]:
    """
    Merges heading-like chunks with following lists or tabular-like items.
    Also merges truncated lists that continue across chunk boundaries.
    """
    merged = []
    i = 0

    while i < len(chunks):
        chunk = chunks[i].strip()
        lines = chunk.splitlines()

        # --- Detect headings (e.g., "E7.2 ACCESSORIES") ---
        is_heading = len(lines) == 1 and (
            re.match(r"^[A-Z][A-Z0-9 ./\-]{2,40}$", lines[0].strip())
            or len(lines[0].split()) <= 5
        )

        # --- Detect if this chunk looks like a list continuation ---
        looks_like_list = any(
            re.match(r"^([•\-–\u2022]|\d+\.)", l.strip()) for l in lines
        ) or all(
            len(l.strip().split()) <= 8 and not re.search(r"[.:;]", l) for l in lines
        )

        merged_chunk = chunk
        merged_count = 0
        j = i + 1

        # --- Merge list items under headings ---
        if is_heading:
            while j < len(chunks):
                next_chunk = chunks[j].strip()
                if re.match(r"^([•\-–\u2022]|\d+\.)", next_chunk) or (
                    len(next_chunk.split()) <= 8 and not re.search(r"[.:;]", next_chunk)
                ):
                    merged_chunk += "\n" + next_chunk
                    merged_count += 1
                    j += 1
                else:
                    break

        # --- Handle truncated lists (continuations) ---
        elif looks_like_list and j < len(chunks):
            next_chunk = chunks[j].strip()
            # If current chunk ends abruptly and next starts with another list line
            if (
                not chunk.endswith((".", ":", ";"))
                and re.match(r"^([•\-–\u2022]|\d+\.)", next_chunk)
            ):
                merged_chunk += "\n" + next_chunk
                merged_count += 1
                j += 1

        if Config.DEBUG and merged_count > 0:
            logger.debug(f"[CHUNK MERGE] Chunk {i} merged with {merged_count} continuation chunks")

        merged.append(merged_chunk)
        i = j if merged_count > 0 else i + 1

    return merged
