"""
CLI entrypoint for Part 4: categorize Silver transactions via Groq
(openai/gpt-oss-120b) and write the Gold layer output.
 
Requires a .env file in the project root containing:
    GROQ_API_KEY=your_key_here
Get a free key at https://console.groq.com/keys — copy .env.example to
.env and fill it in before running this.
 
Usage (run from the project root):
    python scripts/run_gold_pipeline.py
"""
from __future__ import annotations
 
import logging
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.gold.pipeline import run_gold_pipeline  # noqa: E402
 
 
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
 
    output_path = run_gold_pipeline()
    print(f"\n✅ Gold transactions written: {output_path}")
 
 
if __name__ == "__main__":
    main()
