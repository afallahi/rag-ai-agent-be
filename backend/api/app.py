from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware # Import for CORS
from main.pipeline_core import RAGPipeline, generate_response, get_reranker, get_llm

app = FastAPI()
rag = RAGPipeline()
llm = get_llm()
reranker = get_reranker()
history = []

origins = [
    "*", 
    # Example if you knew the exact origin: "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # List of allowed origins
    allow_credentials=True,         # Allow cookies/authorization headers
    allow_methods=["*"],            # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],            # Allow all headers
)

@app.post("/query")
async def query(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}, 400

    query_text = data.get("query")
    history = data.get("history", [])
    response = generate_response(rag, query_text, llm, history, reranker)
    return {"results": [response]}
