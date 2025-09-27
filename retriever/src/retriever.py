import os
from typing import Dict, List

from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

MILVUS_HOST = os.getenv(
    "MILVUS_HOST",
    os.getenv(
        "RETRIEVER_HOST",
        "standalone"))
MILVUS_PORT = os.getenv("MILVUS_PORT", os.getenv("RETRIEVER_PORT", "19530"))
COLLECTION = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class MilvusRetriever:
    def __init__(
            self,
            collection_name: str = COLLECTION,
            embed_model: str = EMBED_MODEL):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.model = SentenceTransformer(embed_model)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        emb = self.model.encode(
            [query], normalize_embeddings=True).astype("float32")[0]
        res = self.collection.search(
            data=[emb],
            anns_field="emb",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["doc_path", "chunk_idx", "text"],
        )
        hits = res[0]
        out = []
        for h in hits:
            out.append({
                "score": float(h.distance),
                "text": h.entity.get("text"),
                "doc_path": h.entity.get("doc_path"),
                "chunk_idx": int(h.entity.get("chunk_idx")),
            })
        return out
