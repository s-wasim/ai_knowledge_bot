import os
import time
import logging
from voyageai import Client as VoyageClient

logger = logging.getLogger(__name__)

BATCH_SIZE = 128
MAX_RETRIES = 1
BACKOFF_SECONDS = 5


def get_voyage_client():
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        return None
    return VoyageClient(api_key=api_key)


def embed_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]] | None:
    client = get_voyage_client()
    if client is None:
        return None

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(1 + MAX_RETRIES):
            try:
                response = client.embed(
                    texts=batch,
                    model="voyage-code-3",
                    input_type="document",
                )
                all_embeddings.extend(response.embeddings)
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.warning(f"Voyage API error (attempt {attempt + 1}): {e}, retrying in {BACKOFF_SECONDS}s")
                    time.sleep(BACKOFF_SECONDS)
                else:
                    logger.error(f"Voyage API failed after {1 + MAX_RETRIES} attempts: {e}")
                    raise

    return all_embeddings


def is_voyage_available() -> bool:
    return os.environ.get("VOYAGE_API_KEY") is not None
