from __future__ import annotations
 
import logging
import os
from typing import Any, Optional
 
from dotenv import load_dotenv
from groq import APIConnectionError, Groq, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
 
from src.rag.vector_store import query_transactions
 
logger = logging.getLogger(__name__)
 
load_dotenv()
 
GROQ_MODEL = "openai/gpt-oss-120b"  # see src/gold/categorizer.py for the deprecation note on llama-3.3-70b-versatile
DEFAULT_N_RESULTS = 15
 
_SYSTEM_PROMPT = """You are a financial assistant answering questions about the user's personal transactions.
 
You are given relevant transactions retrieved from the user's transaction history, followed by their question. Answer ONLY using the transactions provided — never invent amounts, dates, or merchants not present in the context. If the provided transactions don't contain enough information to answer, say so directly instead of guessing.
 
Keep answers concise and specific, citing actual figures (amounts, dates, counts) from the context where relevant.
"""
 
_client: Optional[Groq] = None
 
 
def _get_client() -> Groq:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Create a .env file (copy .env.example) with "
            "GROQ_API_KEY=your_key_here — get a free key at https://console.groq.com/keys."
        )
    _client = Groq(api_key=api_key)
    return _client
 
 
def _format_context(query_result: dict[str, Any]) -> str:
    documents = query_result["documents"][0]
    metadatas = query_result["metadatas"][0]
    distances = query_result["distances"][0]
 
    if not documents:
        return ""
 
    lines = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        relevance = 1 - dist  # cosine distance -> similarity
        lines.append(f"- {doc} [category: {meta.get('category', 'Unknown')}, relevance: {relevance:.2f}]")
    return "\n".join(lines)
 
 
@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_groq(question: str, context: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Relevant transactions:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
 
 
def answer_question(question: str, n_results: int = DEFAULT_N_RESULTS) -> str:
    if not question or not question.strip():
        return "Please enter a question about your transactions."
 
    query_result = query_transactions(question.strip(), n_results=n_results)
    context = _format_context(query_result)
 
    if not context:
        return (
            "I couldn't find any transactions in your data related to that question. "
            "Try uploading a statement or logging some transactions first, then rebuild the index."
        )
 
    try:
        return _call_groq(question.strip(), context)
    except RuntimeError:
        raise  # missing GROQ_API_KEY — surface this as-is, not as a generic failure
    except Exception as exc:  # noqa: BLE001 — never let a chat query crash the caller
        logger.exception("RAG chat query failed")
        return f"Sorry, I couldn't generate an answer right now ({exc}). Please try again."
