"""RAG Project Main Module"""

import logging
import argparse
from main.config import Config
from main.logger_config import setup_logging, log_duration
from main.llm.factory import get_llm_client
from main.retrieval.rerankers.reranker_factory import CohereReranker, BedrockCohereReranker
from main.pipeline_core import RAGPipeline
from main.llm.prompt_builder import build_prompt


MAX_HISTORY_LENGTH = 10


setup_logging(logging.DEBUG if Config.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


@log_duration("Generate Response")
def generate_response(rag_pipeline: RAGPipeline, query_text: str, llm, history: list[tuple[str, str]], reranker=None) -> str:
    """Query global index and generate a response using LLM."""

    response = "The retrieved documents do not provide enough information."
    intent = rag_pipeline.intent_detector.detect(query_text)

    quick_responses = {
        "greeting": "Hello! I'm your Armstrong assistant. Ask me a technical question and I’ll look it up for you.",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "goodbye": "Goodbye! Feel free to come back with more questions anytime.",
        "help": "I can help answer questions about your HVAC questions and Armstrong products. Ask me something specific!",
        "vague": "Could you please rephrase your question or ask something more specific?",
        "empty": None
    }
    
    if intent in quick_responses:
        return quick_responses[intent] or response

    final_docs = rag_pipeline.query_knowledge_base(query_text, reranker)
    if not final_docs:
        return "Sorry, I couldn't find relevant information in the documents."
    
    logger.debug("Retrieved %d chunks for query: '%s'", len(final_docs), query_text)
    context = "\n\n".join(final_docs)
    prompt = build_prompt(context, query_text, history)
    return llm.generate_answer(prompt)


@log_duration("Total Query Time")
def query_and_respond(rag_pipeline: RAGPipeline, query_text: str, llm, history: list[tuple[str, str]], reranker=None) -> None:
    response = generate_response(rag_pipeline, query_text, llm, history, reranker)
    
    print(f"\nAssistant: {response}")
    history.append((query_text, response))

    if len(history) > MAX_HISTORY_LENGTH:
        history[:] = history[-MAX_HISTORY_LENGTH:]


def get_llm(provider: str=None, default_provider: str=None):
    provider = provider or Config.LLM_PROVIDER or "ollama"
    default_provider = default_provider or provider
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


def chat_loop(rag_pipeline: RAGPipeline, llm, history, reranker, default_provider):
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

            query_and_respond(rag_pipeline, query, llm, history, reranker=reranker)
    except KeyboardInterrupt:
        logger.info("\nExiting on user interrupt")


def main():
    """Main"""

    default_provider = Config.LLM_PROVIDER or "ollama"
    llm = get_llm(default_provider, default_provider)
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
    chat_loop(rag_pipeline, llm, history, reranker, default_provider)
        

if __name__ == "__main__":
    main()
