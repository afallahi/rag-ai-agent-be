def build_prompt(context: str, query: str, history: list[tuple[str, str]]) -> str:
    """
    Construct the LLM prompt given context, query, and conversation history.
    Ensures lists, tables, and structured data are preserved fully.
    """
    conversation = ""
    for i, (prev_q, prev_a) in enumerate(history, start=1):
        conversation += f"\nQ{i}: {prev_q}\nA{i}: {prev_a}"

    return (
        "You are a professional HVAC systems consultant. "
        "Use ONLY the information provided in the context below to answer the customer's question. "
        "If the context includes a list, table, or structured data (for example: technical data, materials, or accessories), "
        "list **all** items completely and accurately, without summarizing or omitting any. "
        "If the context does not contain the answer, say 'The context does not provide enough information.' "
        "When listing items, preserve their exact names and units from the source.\n\n"
        f"{conversation}\n\nContext:\n{context}\n\nQuestion: {query}"
    )
