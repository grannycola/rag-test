from datetime import datetime
import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest, Mount


MILVUS_HOST = os.getenv("MILVUS_HOST", "standalone")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION  = os.getenv("COLLECTION_NAME", "rag_chunks")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_DIR    = os.getenv("DATA_DIR", "/data/docs")

with DAG(
    dag_id="rag_ingest_docker",
    start_date=datetime(2025,1,1),
    schedule="@daily",     # или None, если запускать вручную
    catchup=False,
    tags=["rag","milvus"],
) as dag:

    ingest = DockerOperator(
        task_id="ingest_chunks",
        image="rag-ingest:latest",
        api_version="auto",
        docker_url="unix://var/run/docker.sock",
        network_mode="milvus",
        device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
        auto_remove="success",
        environment={
            "MILVUS_HOST": MILVUS_HOST,
            "MILVUS_PORT": MILVUS_PORT,
            "COLLECTION_NAME": COLLECTION,
            "EMBED_MODEL": EMBED_MODEL,
            "DATA_DIR": DATA_DIR,
        },
        mounts=[
            Mount(source="/opt/airflow/hf-cache", target="/opt/hf-cache", type="bind"),
            Mount(source="/opt/airflow/torch-cache",target="/opt/torch-cache", type="bind"),
            Mount(source="/data/docs", target="/data/docs", type="bind", read_only=True),
        ],
    )

    ingest
 