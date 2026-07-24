from __future__ import annotations
 
import os
 
# MUST run before the chromadb/sentence-transformers import below, which
# transitively pulls in torch. On Windows, pandas' bundled Intel MKL
# OpenMP runtime and torch's own bundled OpenMP runtime (both
# libiomp5md.dll) can collide when both load in one process — observed
# as a silent hard crash (exit code -1073741819 / 0xC0000005), not a
# catchable Python exception, when pandas is imported (e.g. in
# indexer.py) before anything that pulls in torch. setdefault(), not a
# hard overwrite, so an explicit environment-level choice isn't clobbered.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
 
# The model is already fully downloaded and cached locally (confirmed by
# an earlier successful run). Without this, huggingface_hub does a HEAD
# request to huggingface.co on EVERY run just to check the cache is still
# current — which is what timed out. This skips that check and loads
# straight from the local cache.
#
# IMPORTANT: if you ever delete the HF cache or move to a new machine,
# temporarily comment this line out (or set HF_HUB_OFFLINE=0 in the
# environment) for one run so it can download fresh, then restore it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
 
from functools import lru_cache
 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
 
from src.config import EMBEDDING_MODEL_NAME
 
 
@lru_cache(maxsize=1)
def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    # cached: loading the model from disk/HF cache is the slow part;
    # every caller in this process shares one instance.
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device="cpu",
        normalize_embeddings=True,  # required for cosine similarity via Chroma's hnsw "cosine" space
    )
 
 
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    ef = get_embedding_function()
    return ef(texts)