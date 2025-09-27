## RAG Test Project

Проект для построения Retrieval-Augmented Generation (RAG) на базе Milvus. Включает сервис ингеста текстов с эмбеддингами, хранилище векторных данных, интеграцию с Airflow и утилиты.

### Состав
- **Ingest (`ingest/`)**: Python-сервис, который
  - читает файлы из каталога, 
  - разбивает их на чанки (LangChain `RecursiveCharacterTextSplitter`, либо fallback),
  - строит эмбеддинги SentenceTransformers,
  - пишет чанки в Milvus, обеспечивая идемпотентность через `doc_hash`.
- **Vector DB (`vector_db/`)**: Milvus (standalone/cluster), управляется через Docker Compose.
- **Airflow (`airflow/`)**: DAG для запуска ингеста (через DockerOperator).
- **Утилиты**: `view_milvus.py` — быстрый просмотр/проверка коллекции.

### Дерево проекта (ключевое)
- `ingest/src/ingest.py` — основная логика ингеста
- `ingest/Dockerfile` — сборка образа ингеста
- `vector_db/docker-compose.yaml` — Milvus и его зависимости
- `airflow/docker-compose.yaml` — стек Airflow
- `airflow/dags/rag_ingest_docker.py` — DAG для запуска ингеста
- `view_milvus.py` — примеры доступа/просмотра Milvus

## Быстрый старт

### 1) Поднять Milvus
```bash
docker compose -f vector_db/docker-compose.yaml -p rag-db up -d
```

Проверьте, что Milvus доступен на `MILVUS_HOST`/`MILVUS_PORT` (см. переменные ниже).

### 2) Подготовить данные
Поместите `.txt`/`.md` файлы в каталог, который будет смонтирован как `/data/docs` в контейнере ингеста.

### 3) Собрать и запустить Ingest
```bash
docker build -t rag-ingest:latest ingest

# Пример запуска (локально без Airflow)
docker run --rm \
  -e MILVUS_HOST=milvus \
  -e MILVUS_PORT=19530 \
  -e COLLECTION_NAME=rag_chunks \
  -e DATA_DIR=/data/docs \
  -e RECREATE_COLLECTION_IF_MISSING_FIELDS=1 \
  -e AUTO_DROP_ON_LOAD_ERROR=1 \
  -e CHUNK_SIZE=1800 \
  -e CHUNK_OVERLAP=200 \
  -v $(pwd)/data/docs:/data/docs:ro \
  --network rag-db_default \
  rag-ingest:latest
```

Примечание: сеть `--network rag-db_default` — это сеть проекта Milvus, созданная `docker compose -p rag-db ...`. Уточните имя через `docker network ls`.

## Переменные окружения (ingest)
- `MILVUS_HOST` (строка, по умолчанию `standalone`) — адрес Milvus
- `MILVUS_PORT` (строка, по умолчанию `19530`) — порт Milvus
- `COLLECTION_NAME` (строка, по умолчанию `rag_chunks`) — имя коллекции
- `EMBED_MODEL` (строка, по умолчанию `sentence-transformers/all-MiniLM-L6-v2`) — модель эмбеддингов
- `DATA_DIR` (строка, по умолчанию `/data/docs`) — каталог с документами
- `LOG_LEVEL` (строка, по умолчанию `INFO`) — уровень логирования
- `RECREATE_COLLECTION_IF_MISSING_FIELDS` (`0`/`1`) — если в существующей коллекции нет нужных полей (например, `doc_hash`), удалить и пересоздать
- `AUTO_DROP_ON_LOAD_ERROR` (`0`/`1`) — при ошибке загрузки коллекции с кодом 2001 (битые сегменты/хранилище) удалить и пересоздать
- `CHUNK_SIZE` (число, символов) — размер чанка для LangChain-сплиттера
- `CHUNK_OVERLAP` (число, символов) — overlap для LangChain-сплиттера

## Идемпотентный ингест
- Для каждого файла вычисляется `doc_hash` (MD5 содержимого). 
- В Milvus используется строковое поле `doc_hash` и scalar-индекс `INVERTED` (Milvus 2.6+), 
  чтобы быстро проверять существование документа перед чанкированием и созданием эмбеддингов.
- Если документ уже есть — он пропускается до чанкинга и до вызова модели.

## Чанкирование
- По умолчанию используется LangChain `RecursiveCharacterTextSplitter` (пакет `langchain-text-splitters`).
- Если пакет недоступен, срабатывает fallback: split по предложениям с лимитом по «словам».
- Размеры настраиваются через `CHUNK_SIZE`/`CHUNK_OVERLAP`.

## Схема коллекции Milvus
Коллекция `rag_chunks` (по умолчанию):
- `id` — INT64, primary, auto_id
- `doc_path` — VARCHAR(512)
- `doc_hash` — VARCHAR(32)
- `chunk_idx` — INT64
- `text` — VARCHAR(8192)
- `emb` — FLOAT_VECTOR(dim) с индексом HNSW (IP)

Примечание: если коллекция была создана ранее без `doc_hash`, включите `RECREATE_COLLECTION_IF_MISSING_FIELDS=1`.

## Работа через Airflow
- В `airflow/dags/rag_ingest_docker.py` используется DockerOperator для запуска контейнера ингеста.
- Важные переменные окружения для оператора: 
  - `MILVUS_HOST`, `MILVUS_PORT`, `COLLECTION_NAME`, `DATA_DIR`
  - `RECREATE_COLLECTION_IF_MISSING_FIELDS=1` и/или `AUTO_DROP_ON_LOAD_ERROR=1` при миграциях/битых сегментах
- Смонтируйте каталог с документами в контейнер ингеста как `/data/docs`.

## Просмотр коллекции
Примерный скрипт: `view_milvus.py`. Можно запускать локально (при наличии `pymilvus`) и проверять документы/количество записей.

## Docker Compose: один файл или несколько
- Небольшие окружения — удобно иметь один compose с профилями.
- Для разделения доменов — держите `vector_db/docker-compose.yaml` отдельно, а ингест запускайте отдельным compose или просто `docker run`.
- Compose-файлы можно объединять оверлеями: `docker compose -f base.yml -f overlay.yml up -d`.

## Kubernetes (когда нужно)
- Для прод-нагрузки, HA, Milvus cluster, HPA, cron-инжест — Kubernetes логичен (Helm/Operator для Milvus + `Deployment`/`CronJob` для сервисов).
- Для локалки и небольших деплоев достаточно Docker Compose.

## Трюблшутинг
- "cannot create index on non-exist field: doc_hash"
  - Коллекция создана без поля `doc_hash`. Включите `RECREATE_COLLECTION_IF_MISSING_FIELDS=1` или вручную дропните коллекцию перед повторным запуском.
- "failed to load field data ... Path does not exist '.../insert_log/..." (код 2001)
  - Проблема с сегментами/хранилищем (S3/MinIO). Проверьте бакет/volume и креды. Если данные не критичны — включите `AUTO_DROP_ON_LOAD_ERROR=1`.
- Производительность проверки дубликатов
  - Пер-файловая проверка по `doc_hash` через индекс `INVERTED` — быстрая и не грузит ОЗУ.

## Разработка
- Образ ингеста собирается из `ingest/Dockerfile`. Torch уже в базовом образе, зависимости ставятся из `requirements.txt`.
- Логи форматируются для удобного чтения в Airflow. 
- Можно переопределить модель эмбеддингов через `EMBED_MODEL`.

## Лицензия
Не указана. Добавьте при необходимости.


