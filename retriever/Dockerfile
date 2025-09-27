FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for sentence-transformers (torch needs some basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY retriever/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY retriever /app/retriever

EXPOSE 8000

CMD ["uvicorn", "retriever.api:app", "--host", "0.0.0.0", "--port", "8000"]


