import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection


MILVUS_HOST = os.getenv("MILVUS_HOST", os.getenv("RETRIEVER_HOST", "standalone"))
MILVUS_PORT = os.getenv("MILVUS_PORT", os.getenv("RETRIEVER_PORT", "19530"))
COLLECTION  = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class MilvusRetriever:
    def __init__(self, collection_name: str = COLLECTION, embed_model: str = EMBED_MODEL):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.model = SentenceTransformer(embed_model)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        emb = self.model.encode([query], normalize_embeddings=True).astype("float32")[0]
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


def _cli():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("query", help="Query text")
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    r = MilvusRetriever()
    res = r.search(args.query, top_k=args.top_k)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()



