import re

def normalize_text(text: str) -> str:
    """
    Normalize text by spacing out structured alphanumeric codes like 'E7.2' or 'E7.2B'.
    Example:
        'E7.2B' -> 'E 7_2 B'
        'E22.2/E22.2B' -> 'E 22_2 / E 22_2 B'
    """
    # Replace patterns like E7.2 or E7.2B with a spaced + underscored version
    pattern = re.compile(r"\b([A-Za-z])(\d+)\.(\d+)([A-Za-z]?)\b")
    text = pattern.sub(lambda m: f"{m.group(1)} {m.group(2)}_{m.group(3)} {m.group(4)}".strip(), text)
    
    # Normalize multiple spaces and make consistent casing
    text = re.sub(r"\s+", " ", text).strip()
    return text

