import re
import logging

logger = logging.getLogger(__name__)

# Map all known Unicode fractions to ASCII
FRACTION_MAP = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

def fix_pdf_symbols(text: str) -> str:
    """
    Normalize special fraction glyphs and other non-ASCII characters
    that Textract often misses or mis-recognizes.
    This works for all PDFs, not just specific ones.
    """
    if not text:
        return text

    # original = text
    replaced = []

    # Replace fraction glyphs
    for glyph, ascii_equiv in FRACTION_MAP.items():
        if glyph in text:
            text = text.replace(glyph, ascii_equiv)
            replaced.append(glyph)

    # Normalize dashes and minus variants
    text = re.sub(r"[‐-–—]", "-", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    if replaced:
        logger.debug(f"[fix_pdf_symbols] Fraction fixes applied: {replaced}")

    return text


def merge_split_tokens(text: str) -> str:
    """
    Merge split tokens like '1' + '¼"' into '1¼"' when Textract separates them.
    This is heuristic-based and works best after symbol normalization.
    """
    # Merge patterns like '1 1/4"' → '1¼"'
    text = re.sub(r"\b1\s+1/4\"", "1¼\"", text)
    text = re.sub(r"\b1\s+1/2\"", "1½\"", text)
    text = re.sub(r"\b3\s+4\"", "¾\"", text)

    return text


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text before chunking to preserve lists, bullets, and structure.
    Works generically for any document with bullet-like structures.
    """
    # Repair malformed PDF text first
    text = fix_pdf_symbols(text)
    text = merge_split_tokens(text)
    # text = recover_fraction_lines(text)

    # Normalize bullets and dashes
    text = (
        text.replace("•", "\n• ")
        .replace("·", "\n• ")
        .replace("–", "-")
        .replace("—", "-")
    )

    # Remove weird hyphenations split across lines (e.g., "Flange-\nkits" → "Flange kits")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Add line breaks between bullets if they’re stuck together
    text = re.sub(r"•\s*(?=[^•])", r"• ", text)
    text = re.sub(r"(?<=\S)•", r"\n•", text)

    # Collapse excessive newlines but keep list separation
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join continuation lines that don't start with a bullet, number, or capital (for paragraph flow)
    text = re.sub(r"\n(?![\s•\dA-Z-])", " ", text)

    # Preserve list formatting (ensure bullets start on new lines)
    text = re.sub(r"([^\n])(\s*•)", r"\1\n\2", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text


def recover_fraction_lines(text: str) -> str:
    """
    Heuristically recover fraction patterns that Textract may miss or split.
    """
    # Recover common patterns missed by Textract
    text = re.sub(r"\b1\s+1/4\"", "1¼\"", text)
    text = re.sub(r"\b1\s+1/2\"", "1½\"", text)
    text = re.sub(r"\b3\s+4\"", "¾\"", text)
    return text

