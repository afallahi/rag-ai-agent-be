def build_prompt(context: str, query: str, history: list[tuple[str, str]]) -> str:
    """
    Construct the LLM prompt given context, query, and conversation history.
    """
    conversation = ""
    for i, (prev_q, prev_a) in enumerate(history, start=1):
        conversation += f"\nQ{i}: {prev_q}\nA{i}: {prev_a}"

    return (
        "You are a professional HVAC systems consultant. "
        "Use ONLY the context below to answer the following customer question.\n"
        "Some content in the context may come from graphs, tables, or images; interpret this information accurately.\n"
        "Answer in a concise, informative paragraph. If the context does not contain the answer, "
        "say 'The context does not provide enough information.'\n\n"
        f"{conversation}\n\nContext:\n{context}\n\nQuestion: {query}"
    )
