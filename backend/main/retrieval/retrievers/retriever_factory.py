from main.retrieval.retrievers.faiss_retriever import FAISSRetriever
from main.retrieval.retrievers.bedrock_retriever import BedrockRetriever

def get_retriever(retriever_type="faiss", force=False):
    retriever_type = retriever_type.lower()
    if retriever_type == "faiss":
        return FAISSRetriever(force=force)
    elif retriever_type == "bedrock":
        return BedrockRetriever()
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
