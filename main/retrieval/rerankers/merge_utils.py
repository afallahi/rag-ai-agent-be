import logging
from main.config import Config

logger = logging.getLogger(__name__)



def merge_adjacent_chunks(top_results, all_chunks, window_size=Config.MERGE_WINDOW_SIZE, proximity_merge=Config.PROXIMITY_MERGE):
    """
    Merge adjacent chunks around selected FAISS results to preserve section completeness.
    Expands merge window to capture full tables/lists.
    """
    merged = []
    used = set()

    # Sort to ensure merging in text order
    ranked_indices = []
    for chunk_text, _ in top_results:
        try:
            idx = all_chunks.index(chunk_text)
            ranked_indices.append(idx)
        except ValueError:
            continue

    ranked_indices = sorted(set(ranked_indices))
    last_end = -1

    for idx in ranked_indices:
        if idx in used:
            continue

        # Expand window
        start = max(0, idx - window_size)
        end = min(len(all_chunks), idx + window_size + 1)

        # If proximity_merge is on, merge close chunks (e.g., within 2 positions)
        if proximity_merge and start <= last_end + 2:
            start = min(start, last_end)
        last_end = end

        neighbors = all_chunks[start:end]
        merged_text = "\n".join(neighbors)

        logger.debug(
            "Merged %d chunks (indices %d–%d) around index %d into one extended section.",
            len(neighbors),
            start,
            end - 1,
            idx
        )

        merged.append(merged_text)
        used.update(range(start, end))

    return merged
