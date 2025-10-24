# RAG PDF Agent
This project builds a Retrieval-Augmented Generation (RAG) system that enables users to ask technical questions based on the content of PDF documents. It extracts text from one or more PDFs, chunks and embeds it, builds a FAISS vector index, and uses a language model to generate contextual responses. The system supports both local LLMs via Ollama (e.g., Mistral) and cloud-hosted models via AWS Bedrock (e.g., Claude).


## Project Goals

- Extract text from PDFs
- Chunk text using LangChain
- Generate embeddings using Sentence Transformers
- Store and query a unified FAISS vector database
- Use a local LLM (Ollama + Mistral) to answer user questions
- LLM-powered intent detection via LangChain
- CLI-based interactive assistant
- Streamlit-based web chat UI
- REST API backend (FastAPI)
- Manifest-based incremental indexing
- Validate each stage with unit tests


## Architecture Diagram

[Architecture Diagram](shared\architecture.mmd)

## Project Structure

```
.
|-- .gitignore
|-- README.md
|-- backend
|   |-- .env
|   |-- api
|   |   `-- app.py
|   |-- main
|   |   |-- chunker
|   |   |   `-- text_chunker.py
|   |   |-- config.py
|   |   |-- embedder
|   |   |   `-- embedder.py
|   |   |-- extractor                       # PDF extractors
|   |   |   |-- chart_ocr_extractor.py
|   |   |   |-- pdf_extractor_base.py
|   |   |   |-- pdf_extractor_factory.py
|   |   |   |-- pdf_extractor_hybrid.py
|   |   |   |-- pdf_extractor_pymupdf.py
|   |   |   `-- pdf_extractor_textract.py
|   |   |-- intent_detector
|   |   |   |-- bedrock_intent_detector.py
|   |   |   |-- intent_detector_base.py
|   |   |   |-- intent_detector_factory.py
|   |   |   `-- ollama_intent_detector.py
|   |   |-- llm
|   |   |   |-- base.py
|   |   |   |-- bedrock_client.py
|   |   |   |-- factory.py
|   |   |   |-- ollama_client.py
|   |   |   `-- prompt_builder.py
|   |   |-- logger_config.py
|   |   |-- pipeline
|   |   |   `-- file_processor.py
|   |   |-- pipeline_core.py        # RAGPipeline entry point
|   |   |-- retrieval               # Vecore Store, retrievers, rerankers
|   |   |   |-- rerankers
|   |   |   |   |-- bedrock_cohere_reranker.py
|   |   |   |   |-- cohere_reranker.py
|   |   |   |   |-- merge_utils.py
|   |   |   |   |-- reranker_base.py
|   |   |   |   `-- reranker_factory.py
|   |   |   |-- retrievers
|   |   |   |   |-- bedrock_retriever.py
|   |   |   |   |-- faiss_retriever.py
|   |   |   |   |-- retriever_base.py
|   |   |   |   `-- retriever_factory.py
|   |   |   `-- vector_store
|   |   |       |-- faiss_indexer.py
|   |   |       |-- index_builder.py
|   |   |       `-- vector_store_manager.py
|   |   `-- utils
|   |       |-- manifest_helper.py
|   |       |-- normalize_tokens.py
|   |       |-- pdf_helper.py
|   |       |-- s3_helper.py
|   |       `-- text_preprocessor.py
|   |-- pipeline.py
|   `-- requirements.txt
|-- frontend
|   |-- chat_app.py
|   `-- requirements.txt
|-- project_structure.txt
|-- sample_pdfs
|   `-- Circulator_E9_2_E9 2B.pdf
|-- shared
|   `-- architecture.mmd
|-- tests
|   |-- __init__.py
|   |-- test_chunker.py
|   |-- test_embedder.py
|   |-- test_llm_ollama.py
|   |-- test_pdf_extractor.py
|   `-- test_vectore_store.py
`-- tools
    `-- generate_structure.py
```

## Setup

### 1. Create Conda environment

```bash
conda create -n rag python=3.11
```


### 2. Activate environment

`conda activate rag`

### 3. Install requirements

`pip install -r backend/requirements.txt`


### 4. Install and Run Ollama (for local LLM Only)
Ollama is used to run local LLMs like Mistral (default) or lighter models like Gemma on low-memory systems.

Download and install [Ollama](https://ollama.com)
Pull and run a model (e.g., `mistral`)

```bash

ollama pull mistral     # For local machine with more memory
ollama pull gemma:2b    # For low-memory systems
ollama run mistral      # or gemma:2b depending on your setup
```

### 5. Configure Environment
Create a `.env` file in the project root to customize:

- LLM provider
- LLM Model
- Reranker provider
- FAISS indexing options

## Running the Project

### CLI Mode

`python backend/pipeline.py`

### Force Reprocessing
By default, the system skips PDFs that already have a processed FAISS index. To force reprocessing of all PDFs:
`python backend/pipeline.py --force`

### Rebuild Index
TBD

### Stop Ollama
`Stop-Process -Name ollama -Force`

## Running the Chat UI (Streamlit)
`streamlit run frontend/chat_app.py`

## Pipeline Steps

| Step | Description                                     | Status       |
| ---- | ----------------------------------------------- | -----------  |
| 1    | PDF Text Extraction (PyMuPDF or Textract)       | ✅ Completed |
| 2    | Text Chunking (LangChain)                       | ✅ Completed |
| 3    | Embedding with SentenceTransformers             | ✅ Completed |
| 4    | Vector Store Setup (FAISS)                      | ✅ Completed |
| 5    | Manifest-Based Incremental Indexing             | ✅ Completed |
| 6    | Relevance Reranker Integration (Cohere/Bedrock) | ✅ Completed |
| 7    | LLM Integration (Ollama or Bedrock)             | ✅ Completed |
| 8    | Full RAG Pipeline                               | ✅ Completed |
| 9    | Streamlit Chat UI                               | ✅ Completed |
| 10   | REST API Backend (FastAPI)                      | In Progress  |


## Running Tests

`pytest`

## Tools Used

- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF parsing
- [LangChain](https://www.langchain.com/) for chunking logic
- [Sentence Transformers](https://www.sbert.net/) for embedding
- [FAISS](https://github.com/facebookresearch/faiss) Facebook AI Similarity Search for vector search
- [Ollama](https://ollama.com/) for running local LLMs like Mistral.
- AWS Bedrock
- [Streamlit](https://streamlit.io/)
- FastAPI for backend API
