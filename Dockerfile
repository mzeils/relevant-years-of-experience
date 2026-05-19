FROM runpod/base:0.6.2-cuda12.1.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf-cache

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install -r requirements.txt

# Bake the model into the image so cold starts don't pay the download cost.
ARG MODEL_NAME=TechWolf/JobBERT-v2
ENV MODEL_NAME=${MODEL_NAME}
RUN python3.11 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

COPY handler.py .

CMD ["python3.11", "-u", "handler.py"]
