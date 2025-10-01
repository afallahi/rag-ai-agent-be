"""RAG Project Main Module"""

import os
import time
import logging
import argparse
from functools import wraps
from main.extractor import pdf_extractor
from main.chunker import text_chunker
from main.embedder import embedder
from main.vector_store import faiss_indexer
from main.config import Config
from main.intent_detector import IntentDetector
from main.logger_config import setup_logging
from main.llm.factory import get_llm_client
from main.vector_store.reranker_factory import CohereReranker, BedrockCohereReranker


MAX_HISTORY_LENGTH = 10
TOP_K_FAISS = 10
TOP_N_RERANK = 4
FAISS_SCORE_THRESHOLD = 0.2


setup_logging(logging.DEBUG if Config.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_DIR = Config.SAMPLE_DIR
DEBUG_OUTPUT_DIR = Config.DEBUG_OUTPUT_DIR
FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")


os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)


def log_duration(name: str):
    """Decorator to log the duration of a function call."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            logger.debug("%s took %.2f sec", name, time.time() - start)
            return result
        return wrapper
    return decorator


def save_debug_outputs(filename: str, chunks: list[str], embeddings: list[list[float]]):
    """Save chunks and embeddings to debug files."""
    # Save chunks
    debug_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.md")
    with open(debug_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"\n--- Chunk {i} ---\n{chunk}\n")
    logger.debug("Chunks saved to: %s", debug_path)

    # Save embeddings
    debug_embed_path = os.path.join(DEBUG_OUTPUT_DIR, f"{filename}.embeddings.txt")
    with open(debug_embed_path, "w", encoding="utf-8") as f:
        for i, emb in enumerate(embeddings, start=1):
            f.write(f"Embedding {i}: {emb}\n")
    logger.debug("Embeddings saved to: %s", debug_embed_path)


def build_prompt(context: str, query: str, history: list[tuple[str, str]]) -> str:
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


@log_duration("Build Global FAISS Index")
def build_global_index(force: bool = False):
    """Extract, chunk, and embed all PDFs and return a global index."""
    if os.path.exists(FAISS_INDEX_PATH) and not force:
        logger.info("Global index already exists. Skipping reprocessing.")
        return faiss_indexer.load_faiss_index(FAISS_INDEX_PATH)

    all_chunks = []
    all_embeddings = []

    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found.")
        return None

    for file in pdf_files:
        file_path = os.path.join(SAMPLE_DIR, file)
        logger.debug("Processing: %s", file)

        text = pdf_extractor.extract_text_from_pdf(file_path)
        if not text.strip():
            logger.warning("No text extracted from %s", file)
            continue

        chunks = text_chunker.chunk_text(text)
        if not chunks:
            logger.warning("No chunks created for %s", file)
            continue

        logger.debug("Created %d chunks from %s", len(chunks), file)
        embeddings = embedder.embed_text_chunks(chunks)
        if not embeddings:
            logger.warning("No embeddings created for %s", file)
            continue

        logger.debug("Generated %d embeddings from %s", len(embeddings), file)
        if Config.DEBUG:
            save_debug_outputs(file, chunks, embeddings)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)

    if not all_chunks or not all_embeddings:
        logger.warning("No data to build global FAISS index.")
        return None

    logger.debug("Building global FAISS index...")
    index = faiss_indexer.build_faiss_index(all_embeddings, all_chunks)
    faiss_indexer.save_faiss_index(index, FAISS_INDEX_PATH)
    logger.debug("Global FAISS index saved to: %s", FAISS_INDEX_PATH)

    return index


@log_duration("FAISS Query + Rerank")
def retrieve_relevant_docs(index, query_text, embedding_model, reranker=None):
    top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedding_model, k=TOP_K_FAISS)
    if not top_chunks:
        return []

    docs = [chunk for chunk, _ in top_chunks if chunk]

    if reranker:
        reranked = reranker.rerank(query_text, docs, top_n=TOP_N_RERANK)
        return [doc for doc, _ in reranked if doc]

    max_score = max(score for _, score in top_chunks)
    if max_score >= FAISS_SCORE_THRESHOLD:
        return docs

    return []


@log_duration("Generate LLM Answer")
def generate_llm_answer(llm, prompt):
    return llm.generate_answer(prompt)


@log_duration("Total Query Time")
def query_and_respond(index, query_text: str, llm, history: list[tuple[str, str]], intent_detector: IntentDetector, embedding_model, reranker=None) -> None:
    """Query global index and generate a response using LLM."""

    response = "The retrieved documents do not provide enough information."
    intent = intent_detector.detect(query_text)

    quick_responses = {
        "greeting": "Hello! I'm your Armstrong assistant. Ask me a technical question and I’ll look it up for you.",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "goodbye": "Goodbye! Feel free to come back with more questions anytime.",
        "help": "I can help answer questions about your HVAC questions and Armstrong products. Ask me something specific!",
        "vague": "Could you please rephrase your question or ask something more specific?",
        "empty": None
    }
    
    if intent in quick_responses:
        response = quick_responses[intent] or response
        if intent == "empty":
            logger.warning("Empty query. Skipping.")
            return
    else:
        final_docs = retrieve_relevant_docs(index, query_text, embedding_model, reranker)
        if final_docs:
            logger.debug("Retrieved %d chunks for query: '%s'", len(final_docs), query_text)
            context = "\n\n".join(final_docs)
            prompt = build_prompt(context, query_text, history)
            response = generate_llm_answer(llm, prompt)

    print(f"\nAssistant: {response}")
    history.append((query_text, response))

    # Limit history length
    if len(history) > MAX_HISTORY_LENGTH:
        history[:] = history[-MAX_HISTORY_LENGTH:]


def get_llm(provider: str, default_provider: str):
    client = get_llm_client(provider=provider)
    if client.is_running():
        return client
    logger.warning("LLM '%s' not running. Falling back to default '%s'.", provider, default_provider)
    return get_llm_client(provider=default_provider)


def get_reranker():
    if Config.RERANK_PROVIDER == "cohere-direct":
        return CohereReranker(api_key=Config.COHERE_API_KEY)
    elif Config.RERANK_PROVIDER == "cohere-bedrock":
        return BedrockCohereReranker(model_id=Config.COHERE_BEDROCK_RERANK_MODEL_ID, region=Config.BEDROCK_REGION)
    return None


@log_duration("Get Embedding Model")
def get_embedding_model():
    return embedder.get_model()


def switch_llm(default_provider: str):
    available_providers = ["ollama", "bedrock"]
    logger.debug("Available LLM providers:")
    for i, p in enumerate(available_providers, start=1):
        print(f"  {i}. {p}")

    choice = input("Select provider by number or name: ").strip().lower()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(available_providers):
            choice = available_providers[idx]
        else:
            logger.debug("Invalid selection. No changes made.")
            return None

    if choice in available_providers:
        return get_llm(choice, default_provider)
    logger.debug("Invalid selection. No changes made.")
    return None


def chat_loop(index, llm, history, intent_detector, embedding_model, reranker, default_provider):
    print("Welcome to Armstrong Chat Assistant!")
    print("Chat started. Type your question below.")
    print("Commands:\n  /reset            - start over\n  /switch           - switch LLM provider\n  /exit             - quit\n")

    try:
        while True:
            query = input("You: ").strip()
            if not query:
                logger.warning("Empty query. Please enter a question.")
                continue

            if query.lower() in ("/exit", "exit", "quit"):
                logger.info("Exiting chat session.")
                break
            elif query.lower() == "/reset":
                history.clear()
                logger.info("Chat history reset.")
                continue
            elif query.lower() == "/switch":
                new_llm = switch_llm(default_provider)
                if new_llm:
                    llm = new_llm
                    print(f"[INFO] Switched to LLM provider '{llm.provider}'.")
                continue

            query_and_respond(index, query, llm, history, intent_detector, embedding_model, reranker=reranker)
    except KeyboardInterrupt:
        logger.info("\nExiting on user interrupt")


def main():
    """Main"""

    default_provider = Config.LLM_PROVIDER or "ollama"
    llm = get_llm(default_provider, default_provider)
    if not llm.is_running():
        logger.error("%s is not running or accessible.", Config.LLM_PROVIDER.capitalize())
        return
    
    intent_detector = IntentDetector()

    parser = argparse.ArgumentParser(description="Run RAG pipeline on sample PDFs")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if FAISS index exists")
    args = parser.parse_args()

    pdf_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.warning("No PDF files found.")
        return
    
    index = build_global_index(force=args.force)
    if index is None:
        logger.warning("Index could not be created or loaded.")
        return
    
    embedding_model = get_embedding_model()
    history = []
    
    reranker = get_reranker()
    chat_loop(index, llm, history, intent_detector, embedding_model, reranker, default_provider)
        

if __name__ == "__main__":
    main()
