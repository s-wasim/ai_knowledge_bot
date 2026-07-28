FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Hugging Face cache lives at a fixed path so the model baked in below is found
# at runtime without a network round-trip.
ENV HF_HOME=/opt/hf

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Doing this at build time means the
# container never downloads weights at request time, and a broken build fails
# here rather than surfacing as a mysterious runtime degradation.
ARG EMBED_MODEL=jinaai/jina-embeddings-v2-base-code
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('${EMBED_MODEL}', trust_remote_code=True, device='cpu')"

# With the weights already present, refuse any runtime download so a cache miss
# is loud instead of slow.
ENV HF_HUB_OFFLINE=1

COPY . .

EXPOSE 8501

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8501"]
