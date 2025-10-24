"""RAG Project Main Module"""

import logging
import argparse
from main.config import Config
from main.logger_config import setup_logging, log_duration
from main.pipeline_core import RAGPipeline, generate_response, get_reranker, get_llm


MAX_HISTORY_LENGTH = 10


setup_logging(logging.DEBUG if Config.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


@log_duration("Total Query Time")
def query_and_respond(rag_pipeline: RAGPipeline, query_text: str, llm, history: list[tuple[str, str]], reranker=None) -> None:
    response = generate_response(rag_pipeline, query_text, llm, history, reranker)
    
    print(f"\nAssistant: {response}")
    history.append((query_text, response))

    if len(history) > MAX_HISTORY_LENGTH:
        history[:] = history[-MAX_HISTORY_LENGTH:]


def chat_loop(rag_pipeline: RAGPipeline, llm, history, reranker):
    print("Welcome to Armstrong Chat Assistant!")
    print("Chat started. Type your question below.")
    print("Commands:\n  /reset            - start over\n  /exit             - quit\n")

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

            query_and_respond(rag_pipeline, query, llm, history, reranker=reranker)
    except KeyboardInterrupt:
        logger.info("\nExiting on user interrupt")


def main():
    """Main"""

    llm = get_llm()
    if not llm.is_running():
        logger.error("%s is not running or accessible.", Config.LLM_PROVIDER.capitalize())
        return
    
    parser = argparse.ArgumentParser(description="Run RAG pipeline on given PDF documents")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if FAISS index exists")
    args = parser.parse_args()
    
    rag_pipeline = RAGPipeline(force_index=args.force)
    if not rag_pipeline.refresh_index():
        logger.warning("No PDFs found to build the index.")
        return
        
    history = []
    reranker = get_reranker()
    chat_loop(rag_pipeline, llm, history, reranker)
        

if __name__ == "__main__":
    main()
