import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.retriever import MilvusRetriever

app = FastAPI(title="RAG Retriever API")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchHit(BaseModel):
    score: float
    text: str
    doc_path: str
    chunk_idx: int


class SearchResponse(BaseModel):
    hits: List[SearchHit]


_retriever = None


def get_retriever() -> MilvusRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MilvusRetriever()
    return _retriever


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    r = get_retriever()
    hits = r.search(req.query, top_k=req.top_k)
    return {"hits": hits}



