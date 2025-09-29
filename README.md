# RAG PDF Agent
This project builds a **Retrieval-Augmented Generation (RAG)** system that enables users to ask technical questions based on the content of PDF documents. It extracts text from one or more PDFs, chunks and embeds it, builds a FAISS vector index, and uses a local LLM (e.g., Mistral via Ollama) to generate contextual responses.



## Project Goals

- Extract text from PDFs
- Chunk text using LangChain
- Generate embeddings using Sentence Transformers
- Store and query a unified FAISS vector database
- Use a local LLM (Ollama + Mistral) to answer user questions
- LLM-powered intent detection via LangChain
- CLI-based interactive assistant
- Streamlit-based web chat UI
- Validate each stage with unit tests


## Architecture Diagram

[Architecture Diagram](./ARCHITECTURE.mmd)

## Project Structure

```
rag-project/
├── sample_pdfs/
├── pipeline.py
├── chat_app.py
├── main/
│ └── extractor.py
│   └── pdf_extractor.py # Step 1: PDF extraction
│ └── chunker/
│   └── text_chunker.py # Step 2: Text chunking
│ └── embedder/
│   └── embedder.py # Step 3: Embedding
│ └── vector_store/
│   └── faiss_indexer.py # Step 4: FAISS vector DB
│ └── llm/
│   └── llm_ollama.py # Step 5: LLM Integration (Ollama)
│ ├── intent_detector.py # LLM-based intent classifier
│ └── config.py
├── tests/
│ └── test_pdf_extractor.py
│ └── test_text_chunker.py
│ └── test_embedder.py
│ └── test_vector_store.py
│ └── test_llm_ollama.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create Conda environment

```bash
conda create -n rag python=3.11
```


### 2. Activate environment

`conda activate rag`

### 3. Install requirements

`pip install -r requirements.txt`


### 4. Install and Run Ollama (for local LLM)
Ollama is used to run local LLMs like Mistral (default) or lighter models like Gemma on low-memory systems.

Download and install [Ollama](https://ollama.com)
Pull and run a model (e.g., `mistral`)

```bash

ollama pull mistral     # For local machine with more memory
ollama pull gemma:2b    # For low-memory systems
ollama run mistral      # or gemma:2b depending on your setup
```

### 5. Configure Environment
Create a `.env` file in the project root to customize values.

- LLM provider
- LLM Model

## Running the Project

`python pipeline.py`

### Force Reprocessing
By default, the system skips PDFs that already have a processed FAISS index. To force reprocessing of all PDFs:
`python pipeline.py --force`

### Stop Ollama
`Stop-Process -Name ollama -Force`

## Running the Chat UI (Streamlit)
`streamlit run chat_app.py`

## Steps

| Step | Description                         | Status          |
| ---- | ----------------------------------- | --------------  | 
| 1    | PDF Text Extraction (PyMuPDF)       | ✅ Completed    |
| 2    | Text Chunking (LangChain)           | ✅ Completed    |
| 3    | Embedding with SentenceTransformers | ✅ Completed    |
| 4    | Vector Store Setup (FAISS)          | ✅ Completed    |
| 5    | LLM Integration (Ollama)            | ✅ Completed    |
| 6    | Full RAG Pipeline                   | ✅ Completed    |
| 7    | Streamlit Chat UI                   | ✅ Completed    |


## Running Tests

`pytest`

## Tools Used

- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) for PDF parsing
- [LangChain](https://www.langchain.com/) for chunking logic
- [Sentence Transformers](https://www.sbert.net/) for embedding
- [FAISS](https://github.com/facebookresearch/faiss) Facebook AI Similarity Search for vector search
- [Ollama](https://ollama.com/) for running local LLMs like Mistral.
- [Streamlit](https://streamlit.io/)
