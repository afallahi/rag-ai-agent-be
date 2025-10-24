import logging
from main.config import Config
from main.embedder import embedder
from main.retrieval.retrievers.retriever_factory import get_retriever
from main.intent_detector.intent_detector_factory import create_intent_detector
from main.llm.prompt_builder import build_prompt
from main.retrieval.rerankers.reranker_factory import CohereReranker, BedrockCohereReranker
from main.llm.factory import get_llm_client


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline Class"""
    def __init__(self, force_index: bool = False, retriever_type="faiss"):
        self.intent_detector = create_intent_detector()
        self.retriever_type = retriever_type or Config.RETRIEVER_TYPE or "faiss"
        self.embedding_model = embedder.get_model()

        self.top_k_faiss = Config.TOP_K_FAISS
        self.top_n_rerank = Config.TOP_N_RERANK
        self.score_threshold = Config.FAISS_SCORE_THRESHOLD
        
        logger.info(f"Initializing RAGPipeline with retriever: {self.retriever_type.upper()}")
        self.retriever = get_retriever(retriever_type, force=force_index)


    def refresh_index(self):
        if hasattr(self.retriever, "index") and self.retriever.index is not None:
            logger.debug("FAISS index is ready.")
            return self.retriever.index
        else:
            logger.info("No FAISS index found or not applicable for retriever type '%s'.", self.retriever_type)
            return None
 

    def query_knowledge_base(self, query_text, reranker=None, top_k=None):
        top_k = top_k or self.top_k_faiss
        try:
            results = self.retriever.retrieve(
                query_text,
                top_k=top_k,
                embedding_model=self.embedding_model,
                reranker=reranker
            )

            if not results:
                print("No results retrieved!")
                return []
       
            return results
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            return []


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


def get_reranker():
    if Config.RERANK_PROVIDER == "cohere-direct":
        return CohereReranker(api_key=Config.COHERE_API_KEY)
    elif Config.RERANK_PROVIDER == "cohere-bedrock":
        return BedrockCohereReranker(model_id=Config.COHERE_BEDROCK_RERANK_MODEL_ID, region=Config.BEDROCK_REGION)
    return None


def get_llm():
    provider = Config.LLM_PROVIDER
    client = get_llm_client(provider=provider)
    if client.is_running():
        return client
    logger.warning("LLM '%s' not running.", provider)
    return get_llm_client(provider=provider)
