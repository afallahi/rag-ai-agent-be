from fastapi import FastAPI, Request
from main.pipeline_core import RAGPipeline, generate_response, get_reranker, get_llm

app = FastAPI()
rag = RAGPipeline()
llm = get_llm()
reranker = get_reranker()
history = []

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    query_text = data.get("query")
    history = data.get("history", [])
    response = generate_response(rag, query_text, llm, history, reranker)
    return {"results": [response]}
