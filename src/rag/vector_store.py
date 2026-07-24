from __future__ import annotations
 
from functools import lru_cache
from typing import Any, Optional
 
import chromadb
from chromadb.api.models.Collection import Collection
 
from src.config import CHROMA_COLLECTION_NAME, VECTOR_STORE_DIR
from src.rag.embeddings import embed_texts, get_embedding_function
 
# Chroma's own bulk write (SQLite + hnswlib, native code) is chunked
# rather than sent as one large call — a defensive mitigation against a
# native crash (0xC0000005) observed on Windows during a ~580-item
# upsert. NOTE: not conclusively proven to be volume-triggered — that
# would need one more isolation step (bulk upsert with precomputed
# embeddings, no chunking) that wasn't run before this was written.
# Chunking is the most defensible fix available without that data point:
# it's safe regardless of the actual native cause and costs nothing at
# this scale. If crashes persist even chunked, that skipped isolation
# test is still the fastest way to find the real boundary.
UPSERT_CHUNK_SIZE = 50
 
 
@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
 
 
def get_transactions_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
 
 
def upsert_transactions(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    if not (len(ids) == len(documents) == len(metadatas)):
        raise ValueError(
            f"ids/documents/metadatas length mismatch: {len(ids)}/{len(documents)}/{len(metadatas)}"
        )
    if not ids:
        return
 
    collection = get_transactions_collection()
 
    for start in range(0, len(ids), UPSERT_CHUNK_SIZE):
        end = start + UPSERT_CHUNK_SIZE
        chunk_ids = ids[start:end]
        chunk_documents = documents[start:end]
        chunk_metadatas = metadatas[start:end]
 
        # Embed on the main thread ourselves, proven safe at full 580-item
        # scale in isolation. Passing embeddings= explicitly means Chroma
        # never invokes the embedding function internally — that internal
        # invocation (not the embedding computation itself) is what
        # crashed in the single-item diagnostic.
        chunk_embeddings = embed_texts(chunk_documents)
 
        collection.upsert(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
        )
 
 
def query_transactions(
    query_text: str,
    n_results: int = 10,
    where: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    collection = get_transactions_collection()
    query_embedding = embed_texts([query_text])[0]
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
 
 
def get_indexed_ids() -> set[str]:
    collection = get_transactions_collection()
    result = collection.get(include=[])
    return set(result["ids"])
 
 
def collection_count() -> int:
    return get_transactions_collection().count()
