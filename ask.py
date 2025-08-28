#!/usr/bin/env python3
import argparse
import os
from typing import List
from rich import print
from rich.markdown import Markdown

import yaml
import chromadb
from chromadb.utils import embedding_functions

try:
    import ollama
except Exception as e:
    ollama = None
    print("[yellow]Warning:[/yellow] Could not import 'ollama' Python package. Install it or use the REST API.")

SYSTEM_PROMPT = """You are a precise, terse technical assistant.
You answer ONLY using the provided CONTEXT. If the context is insufficient, say:
"I don't have enough information in the local knowledge base for that."

Guidelines:
- Prefer the most relevant snippets.
- Include file paths and chunk ranges when citing sources.
- If the question seems actionable (setup steps, commands), list steps clearly.
- If multiple interpretations exist, state assumptions briefly.
"""

def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("chroma_path", "./chroma_db")
    cfg.setdefault("collection", "local_kb")
    cfg.setdefault("model", {"embedder": "BAAI/bge-small-en-v1.5", "llm": "llama3:8b"})
    return cfg

def connect_collection(chroma_path: str, collection_name: str, embedder_model: str):
    client = chromadb.PersistentClient(path=chroma_path)
    st_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedder_model)
    col = client.get_or_create_collection(name=collection_name, embedding_function=st_embed)
    return col

def retrieve_context(collection, query: str, k: int = 6):
    res = collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    items = []
    for i, (d, m) in enumerate(zip(docs, metas)):
        items.append({
            "rank": i + 1,
            "text": d,
            "path": m.get("path"),
            "start": m.get("start"),
            "end": m.get("end"),
            "distance": distances[i] if i < len(distances) else None,
        })
    return items

def build_context_block(items: List[dict]) -> str:
    parts = []
    for it in items:
        header = f"[{it['rank']}] {it['path']}:{it['start']}–{it['end']}"
        parts.append(header + "\n" + it["text"])
    return "\n\n-----\n\n".join(parts)

def call_llm_ollama(model: str, system_prompt: str, user_prompt: str) -> str:
    if ollama is None:
        raise RuntimeError("The 'ollama' Python package is missing. Install it via 'pip install ollama'.")
    r = ollama.chat(model=model, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return r["message"]["content"]

def main():
    parser = argparse.ArgumentParser(description="Ask your local KB (RAG over Chroma + Ollama).")
    parser.add_argument("query", nargs="+", help="Your question")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks to retrieve")
    parser.add_argument("--model", "-m", default=None, help="Override Ollama model name")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved sources")
    args = parser.parse_args()

    query = " ".join(args.query)
    cfg = load_config(args.config)

    collection = connect_collection(cfg["chroma_path"], cfg["collection"], cfg["model"]["embedder"])
    items = retrieve_context(collection, query, k=args.k)

    if not items:
        print("[red]No results in local knowledge base.[/red] Try re-indexing or adjusting your query.")
        return

    context_block = build_context_block(items)
    prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context_block}\n\nAnswer:"

    model = args.model or cfg["model"]["llm"]
    try:
        answer = call_llm_ollama(model, SYSTEM_PROMPT, prompt)
    except Exception as e:
        print(f"[red]LLM error:[/red] {e}")
        return

    print()
    print(Markdown(answer))

    if args.show_sources:
        print("\n[bold]Sources:[/bold]")
        for it in items:
            print(f"- [{it['rank']}] {it['path']}:{it['start']}–{it['end']}")

if __name__ == "__main__":
    main()
