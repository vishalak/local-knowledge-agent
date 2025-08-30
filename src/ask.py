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
    parser.add_argument("query", nargs="*", help="Your question (omit for interactive mode)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks to retrieve")
    parser.add_argument("--model", "-m", default=None, help="Override Ollama model name")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved sources")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive conversation mode")
    parser.add_argument("--clear-history", action="store_true", help="Clear conversation history")
    parser.add_argument("--show-history", action="store_true", help="Show conversation summary")
    parser.add_argument("--stream", action="store_true", help="Enable streaming responses")
    args = parser.parse_args()

    # Handle query input
    if args.query:
        query = " ".join(args.query)
        interactive_mode = False
    elif args.interactive:
        interactive_mode = True
        query = None
    else:
        # If no query provided and not explicitly interactive, ask for one
        interactive_mode = False
        query = input("Enter your question: ").strip()
        if not query:
            print("[red]No question provided.[/red]")
            return
    
    kb = KnowledgeBase(config_path=args.config)
    
    # Handle special commands
    if args.clear_history:
        kb.clear_conversation()
        print("[green]Conversation history cleared.[/green]")
        return
    
    if args.show_history:
        summary = kb.get_conversation_summary()
        if summary:
            print(f"[cyan]Conversation Summary:[/cyan]")
            print(f"Session ID: {summary['session_id']}")
            print(f"Total exchanges: {summary['total_exchanges']}")
            print(f"Duration: {summary['duration']}")
            if summary['topics']:
                print(f"Main topics: {', '.join(summary['topics'])}")
        else:
            print("[yellow]No conversation history available.[/yellow]")
        return
    
    # Interactive conversation mode
    if interactive_mode:
        print("[cyan]Interactive Knowledge Base Chat[/cyan]")
        print("Type 'quit', 'exit', or press Ctrl+C to exit")
        print("Type '/clear' to clear conversation history")
        print("Type '/history' to show conversation summary")
        print("Type '/sources' to toggle source display")
        print("Type '/stream' to toggle streaming mode")
        print("---")
        
        show_sources_in_interactive = args.show_sources
        use_streaming = args.stream
        
        try:
            while True:
                try:
                    user_input = input("\n[bold cyan]❯[/bold cyan] ").strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    
                    if user_input == '/clear':
                        kb.clear_conversation()
                        continue
                    
                    if user_input == '/history':
                        summary = kb.get_conversation_summary()
                        if summary:
                            print(f"[dim]Session: {summary['session_id']} | "
                                  f"Exchanges: {summary['total_exchanges']} | "
                                  f"Duration: {summary['duration']}[/dim]")
                        else:
                            print("[yellow]No conversation history[/yellow]")
                        continue
                    
                    if user_input == '/sources':
                        show_sources_in_interactive = not show_sources_in_interactive
                        print(f"[dim]Source display: {'ON' if show_sources_in_interactive else 'OFF'}[/dim]")
                        continue
                    
                    if user_input == '/stream':
                        use_streaming = not use_streaming
                        print(f"[dim]Streaming mode: {'ON' if use_streaming else 'OFF'}[/dim]")
                        continue
                    
                    # Process the query
                    if use_streaming:
                        result = kb.ask_with_context(user_input, k=args.k, stream=True)
                        
                        # Show sources if requested
                        if show_sources_in_interactive and result['context_docs']:
                            print("\n[dim]Sources:[/dim]")
                            for item in result['context_docs'][:3]:  # Show top 3 sources
                                source_info = f"  [{item['rank']}] {item['path']}:{item['start']}–{item['end']}"
                                if item.get('distance'):
                                    source_info += f" (dist: {item['distance']:.3f})"
                                print(f"[dim]{source_info}[/dim]")
                        
                        # Stream the response
                        print("\n", end="", flush=True)
                        full_response = ""
                        try:
                            for chunk in result['response_stream']:
                                print(chunk, end="", flush=True)
                                full_response += chunk
                            print()  # New line after streaming
                            
                            # Finalize the streaming response in chat history
                            kb.finalize_streaming_response(
                                user_input, 
                                full_response, 
                                result['context_docs'],
                                result['enhanced_query'],
                                result['start_time']
                            )
                        except Exception as e:
                            print(f"\n[red]Streaming error: {e}[/red]")
                    else:
                        result = kb.ask_with_context(user_input, k=args.k, stream=False)
                        
                        # Show sources if requested
                        if show_sources_in_interactive and result['context_docs']:
                            print("\n[dim]Sources:[/dim]")
                            for item in result['context_docs'][:3]:  # Show top 3 sources
                                source_info = f"  [{item['rank']}] {item['path']}:{item['start']}–{item['end']}"
                                if item.get('distance'):
                                    source_info += f" (dist: {item['distance']:.3f})"
                                print(f"[dim]{source_info}[/dim]")
                        
                        # Show response
                        print(f"\n{result['response']}")
                    
                except KeyboardInterrupt:
                    print("\n[yellow]Interrupted[/yellow]")
                    break
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            print("\n[yellow]Goodbye![/yellow]")
        
        return
    
    # Single-shot mode
    if not query:
        print("[red]No question provided.[/red]")
        return
    
    # Use conversational interface for single questions too
    if args.stream:
        result = kb.ask_with_context(query, k=args.k, stream=True)
        
        if not result['context_docs']:
            print("[red]No results in local knowledge base.[/red] Try re-indexing or adjusting your query.")
            return

        if args.show_sources:
            print(Markdown("--- \n*SOURCES*"))
            for item in result['context_docs']:
                # Basic source info
                source_info = f"[{item['rank']}] {item['path']}:{item['start']}–{item['end']}"
                if item.get('distance'):
                    source_info += f" (distance: {item['distance']:.3f})"
                
                # Add reranking scores if available
                if item.get('bm25_score'):
                    source_info += f" (BM25: {item['bm25_score']:.3f})"
                if item.get('hybrid_score'):
                    source_info += f" (hybrid: {item['hybrid_score']:.3f})"
                
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

        # Stream the response
        full_response = ""
        try:
            for chunk in result['response_stream']:
                print(chunk, end="", flush=True)
                full_response += chunk
            print()  # New line after streaming
            
            # Finalize the streaming response in chat history
            kb.finalize_streaming_response(
                query, 
                full_response, 
                result['context_docs'],
                result['enhanced_query'],
                result['start_time']
            )
        except Exception as e:
            print(f"[red]Streaming error: {e}[/red]")
    else:
        result = kb.ask_with_context(query, k=args.k, stream=False)
        
        if not result['context_docs']:
            print("[red]No results in local knowledge base.[/red] Try re-indexing or adjusting your query.")
            return

        if args.show_sources:
            print(Markdown("--- \n*SOURCES*"))
            for item in result['context_docs']:
                # Basic source info
                source_info = f"[{item['rank']}] {item['path']}:{item['start']}–{item['end']}"
                if item.get('distance'):
                    source_info += f" (distance: {item['distance']:.3f})"
                
                # Add reranking scores if available
                if item.get('bm25_score'):
                    source_info += f" (BM25: {item['bm25_score']:.3f})"
                if item.get('hybrid_score'):
                    source_info += f" (hybrid: {item['hybrid_score']:.3f})"
                
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

        # Show the response
        print(Markdown(result['response']))

if __name__ == "__main__":
    main()
