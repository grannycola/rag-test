from typing import List, Optional

import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Reduce noisy client logs that may use %-style message templates
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("llm")

app = FastAPI(title="RAG LLM API")


class SearchHit(BaseModel):
    score: float
    text: str
    doc_path: str
    chunk_idx: int


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    model: str = Field(default=os.getenv("OLLAMA_MODEL", "llama3:8b"))
    temperature: float = Field(1e-5, ge=0.0, le=1.0)
    max_tokens: int = Field(512, ge=64, le=4096)


class AnswerResponse(BaseModel):
    answer: str
    hits: List[SearchHit]
    model: Optional[str] = None


_http_client: httpx.Client | None = None
_ollama_base: str | None = None


def get_retriever_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        base_url = os.getenv("RETRIEVER_URL", "http://localhost:8001")
        _http_client = httpx.Client(base_url=base_url, timeout=30.0)
    return _http_client


def get_ollama_base() -> str:
    global _ollama_base
    if _ollama_base is None:
        _ollama_base = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    return _ollama_base


def build_rag_messages(query: str, hits: List[SearchHit]) -> List[dict]:
    sources = []
    for h in hits:
        sources.append(f"[{h.doc_path}#chunk{h.chunk_idx}] {h.text}")
    context = "\n\n".join(sources)
    system_prompt = (
        "You are a concise assistant. Answer the user's question using only the provided context. "
        "If the answer is not in the context, say that you don't know. "
        "Cite sources in square brackets like [path#chunk]."
    )
    user_prompt = (
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    try:
        logger.info(f"Generating answer for query: {req.query}")
        retriever_client = get_retriever_client()
        resp = retriever_client.post("/search", json={"query": req.query, "top_k": req.top_k})
        resp.raise_for_status()
        data = resp.json()
        raw_hits = data.get("hits", [])
        hits = [SearchHit(**h) for h in raw_hits]

        # If MOCK_LLM=1, return a test answer
        if os.getenv("MOCK_LLM", "0") == "1":
            return {"answer": "Hello, world!", "hits": hits}

        # Build a simple chat payload for Ollama
        messages = build_rag_messages(req.query, hits)
        payload = {
            "model": req.model,
            "options": {"temperature": req.temperature},
            "messages": messages,
            "stream": False,
        }
        base = get_ollama_base()
        r = httpx.post(f"{base}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        content = j.get("message", {}).get("content", "")
        return {"answer": content, "hits": hits}
    except httpx.HTTPError as e:
        logger.exception("Retriever HTTP error")
        raise HTTPException(status_code=502, detail=f"Retriever error: {str(e)}")
    except Exception as e:
        logger.exception("Unhandled error in /answer")
        raise HTTPException(status_code=500, detail=str(e))
