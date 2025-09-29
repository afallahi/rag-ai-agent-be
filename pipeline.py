"""RAG Project Main Module"""

import os
import time
import logging
import argparse
from main.extractor import pdf_extractor
from main.chunker import text_chunker
from main.embedder import embedder
from main.vector_store import faiss_indexer
from main.config import Config
from main.intent_detector import IntentDetector
from main.logger_config import setup_logging
from main.llm.factory import get_llm_client


MAX_HISTORY_LENGTH = 10

setup_logging()
logger = logging.getLogger(__name__)


SAMPLE_DIR = Config.SAMPLE_DIR
DEBUG_OUTPUT_DIR = Config.DEBUG_OUTPUT_DIR
FAISS_INDEX_PATH = os.path.join("faiss_index", "global.index")


os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)


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
        "Answer in a concise, informative paragraph. If the context does not contain the answer, "
        "say 'The context does not provide enough information.'\n\n"
        f"{conversation}\n\nContext:\n{context}\n\nQuestion: {query}"
    )


def detect_intent(text: str) -> str:
    lowered = text.lower().strip()

    if not lowered:
        return "empty"

    greetings = {"hi", "hello", "hey", "good morning", "good afternoon"}
    if any(greet in lowered for greet in greetings):
        return "greeting"
    
    if "thank" in lowered:
        return "thanks"

    if any(kw in lowered for kw in {"bye", "goodbye", "see you"}):
        return "goodbye"

    if any(kw in lowered for kw in {"help", "what can you do", "who are you"}):
        return "help"

    # if it's short and doesn't look like a question
    if len(lowered) < 6 or not lowered.endswith("?"):
        return "vague"

    return "rag"


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


def query_and_respond(index, query_text: str, llm, history: list[tuple[str, str]], intent_detector: IntentDetector, embedding_model):
    """Query global index and generate a response using LLM."""

    start = time.time()
    intent = intent_detector.detect(query_text)


    if intent == "greeting":
        response = "Hello! I'm your Armstrong assistant. Ask me a technical question and I’ll look it up for you."
    elif intent == "thanks":
        response = "You're welcome! Let me know if you have more questions."
    elif intent == "goodbye":
        response = "Goodbye! Feel free to come back with more questions anytime."
    elif intent == "help":
        response = "I can help answer questions about your HVAC questions and Armstrong products. Ask me something specific!"
    elif intent == "vague":
        response = "Could you please rephrase your question or ask something more specific?"
    elif intent == "empty":
        logger.warning("Empty query. Skipping.")
        return
    else:
        # Use the RAG pipeline
        t1 = time.time()
        top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedding_model, k=4)
        print(f"[DEBUG] FAISS query took {time.time() - t1:.2f} sec")
        if not top_chunks:
            logger.info("No matching chunks found for query: %s", query_text)
            response = "Sorry, I couldn't find relevant information in the documents."
        else:
            max_score = max(score for _, score in top_chunks)
            threshold = 0.2
            if max_score < threshold:
                logger.debug("No relevant chunks found for query: '%s'", query_text)
                response = "I looked through the documents but didn't find anything helpful for that question."
            else:
                logger.debug("Retrieved %d top matching chunks for query: '%s'", len(top_chunks), query_text)
                context = "\n\n".join(chunk for chunk, _ in top_chunks)
                t2 = time.time()
                prompt = build_prompt(context, query_text, history)
                print(f"[DEBUG] Prompt building took {time.time() - t2:.2f} sec")
                t3 = time.time()
                response = llm.generate_answer(prompt)
                print(f"[DEBUG] LLM response took {time.time() - t3:.2f} sec")

    print(f"[DEBUG] Total time to respond: {time.time() - start:.2f} sec")
    print(f"\nAssistant: {response}")
    history.append((query_text, response))

    # Limit history length
    if len(history) > MAX_HISTORY_LENGTH:
        history[:] = history[-MAX_HISTORY_LENGTH:]


def main():
    """Main"""

    llm = get_llm_client()
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
    
    t = time.time()
    embedding_model = embedder.get_model()
    print(f"[DEBUG] embedder.get_model() took {time.time() - t:.2f} sec")
    history = []

    print("[DEBUG] Warming up LLM...")
    t = time.time()
    llm.generate_answer("Hello")  # dummy warm-up call
    print(f"[DEBUG] Warm-up took {time.time() - t:.2f} sec")
    
    try:
        print("Welcome to Armstrong Chat Assistant!")
        print("Chat started. Type your question below.")
        print("Type `/reset` to start over or `/exit` to quit.\n")

        while True:
            query = input("You: ").strip()
            if not query:
                logger.warning("Empty query. Please enter a question.")
                continue
            if query.lower() in ("/exit", "exit", "quit"):
                logger.info("Exiting chat session.")
                break
            if query.lower() == "/reset":
                history.clear()
                logger.info("Chat history reset.")
                continue
            
            query_and_respond(index, query, llm, history, intent_detector, embedding_model)
    except KeyboardInterrupt:
        logger.info("\nExiting on user interrupt")
        

if __name__ == "__main__":
    main()
