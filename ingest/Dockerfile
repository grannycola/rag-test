FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TORCH_HOME=/opt/torch-cache \
    # подсказки NVIDIA runtime (не обязательно, но удобно)
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

# зависимости проекта (без torch — он уже в базе)
COPY requirements.txt /app/requirements.txt
# уберите torch/torchaudio/torchvision из requirements.txt, чтобы не перезатирать CUDA-сборку
RUN sed -i '/^torch/d;/^torchvision/d;/^torchaudio/d' /app/requirements.txt && \
    pip install -r /app/requirements.txt && \
    pip install --no-cache-dir langchain-text-splitters>=0.0.1

# (опционально) прогреть модель, чтобы не тянуть в рантайме
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                  local_dir="/opt/hf-cache/all-MiniLM-L6-v2")
print("Model cached.")
PY

COPY src/ /app/src
ENTRYPOINT ["python", "-m", "src.ingest"]