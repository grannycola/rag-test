import os, re, hashlib
import sys, logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus.exceptions import MilvusException
try:
    # LangChain splitters are split into a separate package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAVE_LC = True
except Exception:
    HAVE_LC = False

MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION  = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR    = os.getenv("DATA_DIR", "/data/docs")
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO").upper()
RECREATE_COLLECTION_IF_MISSING_FIELDS = os.getenv("RECREATE_COLLECTION_IF_MISSING_FIELDS", "0") == "1"
AUTO_DROP_ON_LOAD_ERROR = os.getenv("AUTO_DROP_ON_LOAD_ERROR", "0") == "1"

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
    """Split text into chunks.

    If LangChain is available, uses RecursiveCharacterTextSplitter with
    character-based sizing approximating tokens. Otherwise uses the
    existing simple sentence-based splitter.
    """
    if HAVE_LC:
        chunk_size = int(os.getenv("CHUNK_SIZE", str(max_tokens * 6)))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", str(overlap * 6)))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        for part in splitter.split_text(text):
            yield part
        return

    # Fallback: original sentence-based splitting
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
    if buf:
        yield " ".join(buf)

def ensure_collection(name: str, dim: int):
    """Ensure collection exists with expected schema.
    Returns: (col, has_doc_hash: bool)
    """
    def create_expected_collection():
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="emb", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description="RAG chunks")
        c = Collection(name, schema, consistency_level="Strong")

        # индекс на вектора
        c.create_index(
            field_name="emb",
            index_params={
                "index_type": "HNSW",
                "metric_type": "IP",
                "params": {"M": 24, "efConstruction": 200},
            },
        )

        # индекс на строковый hash (Milvus 2.6: INVERTED)
        try:
            c.create_index(
                field_name="doc_hash",
                index_params={"index_type": "INVERTED", "params": {}}
            )
        except Exception as e:
            logger.warning(f"Could not create INVERTED index on doc_hash: {e}")
        return c

    if utility.has_collection(name):
        col = Collection(name)
        existing_fields = {f.name for f in col.schema.fields}
        has_doc_hash = "doc_hash" in existing_fields

        # если нет поля и разрешена «миграция» — пересоздаём коллекцию
        if not has_doc_hash and RECREATE_COLLECTION_IF_MISSING_FIELDS:
            logger.warning("doc_hash missing → drop & recreate due to RECREATE_COLLECTION_IF_MISSING_FIELDS=1")
            utility.drop_collection(name)
            col = create_expected_collection()
            has_doc_hash = True
        else:
            # если поле есть — создадим индекс, если его ещё нет
            if has_doc_hash:
                try:
                    col.create_index(
                        field_name="doc_hash",
                        index_params={"index_type": "INVERTED", "params": {}}
                    )
                except Exception as e:
                    logger.warning(f"doc_hash index create skipped: {e}")

        # load с авто-дропом при специфической ошибке (опционально)
        try:
            col.load()
        except MilvusException as e:
            if AUTO_DROP_ON_LOAD_ERROR and e.code == 2001:
                logger.error(f"Load failed (2001) → drop & recreate due to AUTO_DROP_ON_LOAD_ERROR=1: {e}")
                utility.drop_collection(name)
                col = create_expected_collection()
                col.load()
                has_doc_hash = True
            else:
                raise
        return col, has_doc_hash

    else:
        col = create_expected_collection()
        col.load()
        return col, True

def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()
    col, has_doc_hash = ensure_collection(COLLECTION, dim)

    batch=[]
    processed_count = 0
    skipped_count = 0
    
    for path, text, file_hash in iter_texts(DATA_DIR):
        # Проверяем существование документа
        exists = []
        try:
            if has_doc_hash:
                exists = col.query(expr=f'doc_hash == "{file_hash}"', output_fields=["doc_hash"], limit=1)
            else:
                # Fallback: проверяем по пути (может не уловить изменения содержимого)
                exists = col.query(expr=f'doc_path == "{path}"', output_fields=["doc_path"], limit=1)
        except Exception as e:
            exists = []
        if exists:
            logger.info(f"Skipping already processed document: {path}")
            skipped_count += 1
            continue
            
        processed_count += 1
        logger.info(f"Processing document: {path}")
        
        for i, ch in enumerate(split_into_chunks(text)):
            record = {"doc_path": path, "chunk_idx": i, "text": ch}
            if has_doc_hash:
                record["doc_hash"] = file_hash
            batch.append(record)
            if len(batch) >= 256:
                embs = model.encode([r["text"] for r in batch], normalize_embeddings=True).astype("float32")
                if has_doc_hash:
                    col.insert([
                        [r["doc_path"] for r in batch],
                        [r["doc_hash"] for r in batch],
                        [r["chunk_idx"] for r in batch],
                        [r["text"] for r in batch],
                        list(embs),
                    ])
                else:
                    col.insert([
                        [r["doc_path"] for r in batch],
                        [r["chunk_idx"] for r in batch],
                        [r["text"] for r in batch],
                        list(embs),
                    ])
                batch.clear()
    if batch:
        embs = model.encode([r["text"] for r in batch], normalize_embeddings=True).astype("float32")
        if has_doc_hash:
            col.insert([
                [r["doc_path"] for r in batch],
                [r["doc_hash"] for r in batch],
                [r["chunk_idx"] for r in batch],
                [r["text"] for r in batch],
                list(embs),
            ])
        else:
            col.insert([
                [r["doc_path"] for r in batch],
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
