import torch  # Prevents OpenMP / DLL initialization collision

from src.rag.indexer import build_or_refresh_index
from src.rag.chat import answer_question
 
print("=== Building/refreshing vector index from Gold ===")
result = build_or_refresh_index()
print(result)
 
print()
print("=== Test chat query ===")
answer = answer_question("What did I spend the most money on?")
print(answer)
