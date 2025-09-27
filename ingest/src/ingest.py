import os, re, hashlib
import sys, logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION  = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR    = os.getenv("DATA_DIR", "/data/docs")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    """Configure root logger to stream to stdout for Airflow to capture."""
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(handler)

setup_logging()
logger = logging.getLogger("ingest")

def iter_texts(root: str):
    p = Path(root)
    for f in p.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".txt",".md"}:
            text = f.read_text(encoding="utf-8", errors="ignore")
            file_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            yield str(f), text, file_hash

def split_into_chunks(text: str, max_tokens=300, overlap=50):
    sents = re.split(r"(?<=[.!?])\s+", text)
    buf, cnt = [], 0
    for s in sents:
        t = s.strip()
        if not t: continue
        tk = len(t.split())
        if cnt + tk <= max_tokens or not buf:
            buf.append(t)
            cnt += tk
        else:
            chunk = " ".join(buf)
            yield chunk
            tail = " ".join(buf)[-overlap*6:]
            buf = [tail, t] if tail else [t]
            cnt = len(" ".join(buf).split())
    if buf: yield " ".join(buf)

def ensure_collection(name: str, dim: int):
    if utility.has_collection(name):
        col = Collection(name)
        # Пытаемся убедиться, что индекс на doc_hash существует для уже созданной коллекции
        try:
            col.create_index(
                field_name="doc_hash",
                index_params={"index_type": "Trie"}
            )
        except Exception:
            pass
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks")
        col = Collection(name, schema, consistency_level="Strong")
        col.create_index(
            field_name="emb",
            index_params={"index_type":"HNSW","metric_type":"IP","params":{"M":24,"efConstruction":200}}
        )
        # Создаём scalar индекс для быстрого поиска по doc_hash (VARCHAR)
        try:
            col.create_index(
                field_name="doc_hash",
                index_params={"index_type": "Trie"}
            )
        except Exception as e:
            logger.warning(f"Could not create index on doc_hash: {e}")
    col.load()
    return col

def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    col = ensure_collection(COLLECTION, dim)

    batch=[]
    processed_count = 0
    skipped_count = 0
    
    for path, text, file_hash in iter_texts(DATA_DIR):
        # Проверяем по индексу: есть ли уже такой doc_hash
        try:
            exists = col.query(
                expr=f'doc_hash == "{file_hash}"',
                output_fields=["doc_hash"],
                limit=1
            )
        except Exception as e:
            exists = []
        if exists:
            logger.info(f"Skipping already processed document: {path}")
            skipped_count += 1
            continue
            
        processed_count += 1
        logger.info(f"Processing document: {path}")
        
        for i, ch in enumerate(split_into_chunks(text)):
            batch.append({"doc_path": path, "doc_hash": file_hash, "chunk_idx": i, "text": ch})
            if len(batch) >= 256:
                embs = model.encode([r["text"] for r in batch], normalize_embeddings=True).astype("float32")
                col.insert([
                    [r["doc_path"] for r in batch],
                    [r["doc_hash"] for r in batch],
                    [r["chunk_idx"] for r in batch],
                    [r["text"] for r in batch],
                    list(embs),
                ])
                batch.clear()
    if batch:
        embs = model.encode([r["text"] for r in batch], normalize_embeddings=True).astype("float32")
        col.insert([
            [r["doc_path"] for r in batch],
            [r["doc_hash"] for r in batch],
            [r["chunk_idx"] for r in batch],
            [r["text"] for r in batch],
            list(embs),
        ])
    col.flush()
    col.release()
    col.load()
    logger.info(f"Ingest done. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    main()
