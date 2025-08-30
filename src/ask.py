#!/usr/bin/env python3
import argparse
from typing import List
from rich import print
from rich.markdown import Markdown
from kb import KnowledgeBase

try:
    import ollama
except ImportError:
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
    
    kb = KnowledgeBase(config_path=args.config)
    items = kb.retrieve_context(query, k=args.k)

    if not items:
        print("[red]No results in local knowledge base.[/red] Try re-indexing or adjusting your query.")
        return

    if args.show_sources:
        print(Markdown("--- \n*SOURCES*"))
        for item in items:
            # Basic source info
            source_info = f"[{item['rank']}] {item['path']}:{item['start']}–{item['end']} (distance: {item['distance']:.3f})"
            
            # Add metadata info if available
            if 'filename' in item and item.get('file_size_kb'):
                source_info += f" | {item['file_size_kb']}KB"
            if item.get('is_code_file'):
                source_info += " | CODE"
            elif item.get('is_documentation'):
                source_info += " | DOCS"
            if item.get('modified_date'):
                # Show just the date part
                mod_date = item['modified_date'][:10] if len(item['modified_date']) >= 10 else item['modified_date']
                source_info += f" | Modified: {mod_date}"
            
            print(source_info)
        print(Markdown("---"))

    context_block = build_context_block(items)
    prompt = f"QUESTION:\n{query}\n\nCONTEXT:\n{context_block}\n\nAnswer:"

    model = args.model or kb.cfg["model"]["llm"]
    try:
        answer = call_llm_ollama(model, SYSTEM_PROMPT, prompt)
        print(Markdown(answer))
    except RuntimeError as e:
        print(f"[red]Error:[/red] {e}")
    except Exception as e:
        print(f"[red]An unexpected error occurred:[/red] {e}")

if __name__ == "__main__":
    main()
