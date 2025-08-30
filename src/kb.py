#!/usr/bin/env python3
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm
from rich import print

import chromadb
from chromadb.utils import embedding_functions
from document_processors import extract_text_from_file
from code_splitters import get_code_aware_chunks
from semantic_chunker import get_semantic_chunks
from metadata_extractor import extract_all_metadata
from query_transformer import transform_query
from reranker import create_reranker
from chat_history import create_chat_history

STATE_DIR = ".kb_state"
STATE_FILE = "state.json"

class KnowledgeBase:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.collection = self._connect_collection()
        self.state_path = Path(STATE_DIR) / STATE_FILE
        self.state = self._load_state()
        
        # Initialize reranker
        rerank_config = self.cfg.get("retrieval", {})
        self.reranker = create_reranker(
            method=rerank_config.get("rerank_method", "bm25"),
            config=rerank_config
        )
        
        # Initialize chat history if enabled
        chat_config = self.cfg.get("chat", {})
        self.chat_enabled = chat_config.get("enabled", True)
        if self.chat_enabled:
            self.chat_history = create_chat_history(
                max_history=chat_config.get("max_history", 10)
            )
        else:
            self.chat_history = None

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Basic validation
        assert "source_dir" in cfg and cfg["source_dir"], "source_dir is required in config.yaml"
        cfg.setdefault("chroma_path", "./chroma_db")
        cfg.setdefault("collection", "local_kb")
        cfg.setdefault("include_extensions", [".md", ".txt", ".py", ".pdf", ".docx", ".doc", ".cs"])
        cfg.setdefault("excludes", [".git", "node_modules", ".venv", "__pycache__", "build", "dist"])
        cfg.setdefault("chunk", {"size": 1200, "overlap": 200, "semantic": True})
        cfg.setdefault("metadata", {"generate_summaries": False, "extract_advanced": True})
        cfg.setdefault("retrieval", {"query_transform": "hyde", "max_results": 10, "rerank_method": "bm25", "rerank_max_results": 5})
        cfg.setdefault("chat", {"enabled": True, "max_history": 10, "use_context": True, "max_context_exchanges": 3})
        cfg.setdefault("model", {"embedder": "BAAI/bge-small-en-v1.5", "llm": "llama3:8b"})
        return cfg

    def _connect_collection(self):
        client = chromadb.PersistentClient(path=self.cfg["chroma_path"])
        st_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.cfg["model"]["embedder"])
        col = client.get_or_create_collection(name=self.cfg["collection"], embedding_function=st_embed)
        return col

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"files": {}}
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _sha256_bytes(self, b: bytes) -> str:
        h = hashlib.sha256()
        h.update(b)
        return h.hexdigest()

    def _file_should_include(self, path: Path) -> bool:
        include_exts = self.cfg["include_extensions"]
        exclude_names = self.cfg["excludes"]
        for part in path.parts:
            if part in exclude_names:
                return False
        return path.suffix.lower() in set([e.lower() for e in include_exts])

    def _iter_files(self):
        root = Path(self.cfg["source_dir"]).expanduser()
        exclude_names = self.cfg["excludes"]
        include_exts = self.cfg["include_extensions"]
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in exclude_names]
            for fn in filenames:
                p = Path(dirpath) / fn
                if self._file_should_include(p):
                    yield p

    def _chunk_text(self, text: str, file_path: Path = None) -> List[Tuple[int, int, str]]:
        """Chunk text, using code-aware or semantic splitting when possible."""
        if file_path:
            # Try code-aware chunking first for code files
            code_chunks = get_code_aware_chunks(file_path, text, self.cfg["chunk"]["size"])
            if code_chunks is not None:
                return code_chunks
        
        # Try semantic chunking if enabled and not a code file
        if self.cfg["chunk"].get("semantic", False):
            semantic_chunks = get_semantic_chunks(text, self.cfg["chunk"]["size"])
            if semantic_chunks is not None:
                return semantic_chunks
        
        # Fall back to regular chunking
        size = self.cfg["chunk"]["size"]
        overlap = self.cfg["chunk"]["overlap"]
        if size <= 0:
            return [(0, len(text), text)]
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            chunk = text[start:end]
            chunks.append((start, end, chunk))
            if end == n:
                break
            start = end - overlap if end < n else end
        return chunks

    def _delete_file_docs(self, file_path: str):
        self.collection.delete(where={"path": file_path})

    def _upsert_file(self, file_path: Path):
        # Use the new document processor to extract text
        text = extract_text_from_file(file_path)
        if text is None:
            return None

        b = file_path.read_bytes()
        chunks = self._chunk_text(text, file_path)
        file_hash = self._sha256_bytes(b)
        mtime = int(file_path.stat().st_mtime)

        # Extract advanced metadata if enabled
        file_metadata = {}
        if self.cfg["metadata"].get("extract_advanced", True):
            file_metadata = extract_all_metadata(
                file_path, 
                text, 
                generate_summaries=self.cfg["metadata"].get("generate_summaries", False),
                llm_model=self.cfg["model"]["llm"]
            )

        ids, docs, metas = [], [], []
        from uuid import uuid4
        for (s, e, c) in chunks:
            ids.append(str(uuid4()))
            docs.append(c)
            
            # Create metadata for this chunk
            chunk_metadata = {
                "path": str(file_path), 
                "start": s, 
                "end": e, 
                "sha256": file_hash, 
                "mtime": mtime
            }
            
            # Add file-level metadata to each chunk
            chunk_metadata.update(file_metadata)
            
            metas.append(chunk_metadata)

        if docs:
            max_batch_size = 5000
            for i in range(0, len(docs), max_batch_size):
                batch_ids = ids[i:i + max_batch_size]
                batch_docs = docs[i:i + max_batch_size]
                batch_metas = metas[i:i + max_batch_size]
                self.collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

        return {"sha256": file_hash, "mtime": mtime, "count": len(docs), "metadata": file_metadata}

    def build(self):
        print("[cyan]Starting fresh build...[/cyan]")
        try:
            result = self.collection.get()
            if result['ids']:
                self.collection.delete(ids=result['ids'])
                print(f"[yellow]Deleted {len(result['ids'])} existing documents[/yellow]")
        except Exception as e:
            print(f"[yellow]Collection appears to be empty or new: {e}[/yellow]")

        self.state = {"files": {}}
        files = list(self._iter_files())
        
        # Process files in parallel
        max_workers = min(4, len(files))  # Limit workers to avoid overwhelming the system
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {executor.submit(self._upsert_file, p): p for p in files}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Indexing", unit="file"):
                file_path = future_to_file[future]
                try:
                    info = future.result()
                    if info:
                        self.state["files"][str(file_path)] = info
                except Exception as e:
                    print(f"[red]Error processing {file_path}: {e}[/red]")

        self._save_state()
        print(f"[green]Build complete.[/green] Indexed {len(self.state['files'])} files.")

    def update(self):
        current_files = set()
        added, changed, unchanged = 0, 0, 0

        for p in tqdm(list(self._iter_files()), desc="Scanning", unit="file"):
            current_files.add(str(p))
            b = p.read_bytes()
            sha = self._sha256_bytes(b)

            prev = self.state.get("files", {}).get(str(p))
            if prev and prev.get("sha256") == sha:
                unchanged += 1
                continue

            self._delete_file_docs(str(p))
            info = self._upsert_file(p)
            if info:
                self.state["files"][str(p)] = info
                if prev is None:
                    added += 1
                else:
                    changed += 1

        removed_files = [fp for fp in list(self.state.get("files", {}).keys()) if fp not in current_files]
        for fp in removed_files:
            self._delete_file_docs(fp)
            del self.state["files"][fp]

        self._save_state()
        print(f"[green]Update complete.[/green] Files - added: {added}, changed: {changed}, removed: {len(removed_files)}, unchanged: {unchanged}.")

    def retrieve_context(self, query: str, k: int = 6):
        """
        Retrieve context using enhanced query transformation techniques.
        
        Args:
            query: The user's query
            k: Number of results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Apply query transformation if configured
        transform_method = self.cfg["retrieval"].get("query_transform", "none")
        
        if transform_method != "none":
            print(f"[cyan]Applying {transform_method} query transformation...[/cyan]")
            transformation = transform_query(
                query, 
                method=transform_method,
                llm_model=self.cfg["model"]["llm"]
            )
            
            # Use transformed queries for search
            search_queries = transformation["transformed_queries"]
            
            if transform_method == "hyde" and "hyde_document" in transformation:
                print(f"[dim]HyDE document: {transformation['hyde_document'][:100]}...[/dim]")
        else:
            search_queries = [query]
        
        # Collect results from all transformed queries
        all_results = []
        max_results = self.cfg["retrieval"].get("max_results", 10)
        
        for i, search_query in enumerate(search_queries):
            try:
                # Adjust k based on number of queries to ensure we get enough total results
                query_k = min(k if i == 0 else k//2, max_results)
                
                res = self.collection.query(
                    query_texts=[search_query], 
                    n_results=query_k, 
                    include=["documents", "metadatas", "distances"]
                )
                
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                distances = res.get("distances", [[]])[0]
                
                for j, (d, m) in enumerate(zip(docs, metas)):
                    result_item = {
                        "rank": len(all_results) + 1,
                        "text": d,
                        "path": m.get("path"),
                        "start": m.get("start"),
                        "end": m.get("end"),
                        "distance": distances[j] if j < len(distances) else None,
                        "query_used": search_query,
                        "query_index": i
                    }
                    
                    # Add all metadata from the document
                    result_item.update(m)
                    all_results.append(result_item)
                    
            except Exception as e:
                print(f"[yellow]Warning: Query '{search_query}' failed: {e}[/yellow]")
                continue
        
        # Remove duplicates based on document path and chunk position
        seen = set()
        unique_results = []
        for item in all_results:
            key = (item.get("path"), item.get("start"), item.get("end"))
            if key not in seen:
                seen.add(key)
                unique_results.append(item)
        
        # Apply reranking if configured
        rerank_method = self.cfg["retrieval"].get("rerank_method", "none")
        rerank_max = self.cfg["retrieval"].get("rerank_max_results", k)
        
        if rerank_method != "none" and unique_results:
            print(f"[cyan]Applying {rerank_method} reranking...[/cyan]")
            
            # Prepare documents for reranking (convert to expected format)
            docs_for_rerank = []
            for item in unique_results:
                doc_dict = {
                    "content": item.get("text", ""),
                    "distance": item.get("distance", 1.0),
                    **item  # Include all original metadata
                }
                docs_for_rerank.append(doc_dict)
            
            # Apply reranking
            reranked_docs = self.reranker.rerank_documents(
                query=query,  # Use original query for reranking
                documents=docs_for_rerank,
                max_results=rerank_max
            )
            
            # Convert back to original format
            final_results = []
            for i, doc in enumerate(reranked_docs[:k]):
                result_item = {
                    "rank": i + 1,
                    "text": doc.get("content", ""),
                    "path": doc.get("path"),
                    "start": doc.get("start"),
                    "end": doc.get("end"),
                    "distance": doc.get("distance"),
                    "query_used": doc.get("query_used"),
                    "query_index": doc.get("query_index"),
                }
                
                # Add reranking scores if available
                if "bm25_score" in doc:
                    result_item["bm25_score"] = doc["bm25_score"]
                if "semantic_score" in doc:
                    result_item["semantic_score"] = doc["semantic_score"]
                if "hybrid_score" in doc:
                    result_item["hybrid_score"] = doc["hybrid_score"]
                
                # Add all other metadata
                for key, value in doc.items():
                    if key not in result_item and key != "content":
                        result_item[key] = value
                
                final_results.append(result_item)
        else:
            # Sort by distance (best matches first) and limit results
            unique_results.sort(key=lambda x: x.get("distance", float('inf')))
            final_results = unique_results[:k]
            
            # Re-rank the results
            for i, item in enumerate(final_results):
                item["rank"] = i + 1
        
        return final_results

    def ask_with_context(self, query: str, k: int = 6) -> Dict[str, Any]:
        """
        Ask a question with conversation context support.
        
        Args:
            query: The user's question
            k: Number of context documents to retrieve
            
        Returns:
            Dictionary with response, context documents, and metadata
        """
        start_time = time.time()
        
        # Enhance query with conversation context if chat is enabled
        enhanced_query = query
        if self.chat_enabled and self.chat_history:
            chat_config = self.cfg.get("chat", {})
            if chat_config.get("use_context", True):
                enhanced_query = self.chat_history.enhance_query_with_context(
                    query, 
                    max_context_exchanges=chat_config.get("max_context_exchanges", 3)
                )
                
                if enhanced_query != query:
                    print("[cyan]Enhanced query with conversation context[/cyan]")
        
        # Retrieve relevant context using the enhanced query
        context_docs = self.retrieve_context(enhanced_query, k=k)
        
        # Generate response using LLM
        response = self._generate_response(query, context_docs, enhanced_query)
        
        # Add to chat history if enabled
        if self.chat_enabled and self.chat_history:
            self.chat_history.add_exchange(
                user_query=query,
                assistant_response=response,
                context_docs=context_docs,
                metadata={
                    "enhanced_query": enhanced_query,
                    "num_context_docs": len(context_docs),
                    "response_time": time.time() - start_time
                }
            )
        
        return {
            "response": response,
            "context_docs": context_docs,
            "enhanced_query": enhanced_query,
            "conversation_id": self.chat_history.session_id if self.chat_history else None,
            "response_time": time.time() - start_time
        }
    
    def _generate_response(self, original_query: str, context_docs: List[Dict], 
                          enhanced_query: str) -> str:
        """
        Generate response using LLM based on context documents.
        
        Args:
            original_query: The user's original question
            context_docs: Retrieved context documents
            enhanced_query: Query enhanced with conversation context
            
        Returns:
            Generated response
        """
        # If no context found, return a helpful message
        if not context_docs:
            return "I don't have enough information in the local knowledge base to answer that question."
        
        # Build context string from retrieved documents
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):  # Limit to top 5 documents
            context_parts.append(f"Source {i}: {doc.get('text', '')}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        conversation_context = ""
        if self.chat_enabled and self.chat_history and self.chat_history.history:
            conversation_context = f"""
Conversation context:
{self.chat_history.get_conversation_context(include_responses=True, max_exchanges=2)}

"""
        
        prompt = f"""{conversation_context}Based on the following context documents, please answer the user's question.

Context:
{context_text}

Question: {original_query}

Please provide a helpful and accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so."""
        
        # For now, return a simple response (in a real implementation, this would call an LLM)
        # This is a placeholder since we don't want to require Ollama for basic functionality
        try:
            # Attempt to use LLM if available
            import requests
            
            llm_model = self.cfg["model"]["llm"]
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": llm_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "No response generated.")
            else:
                raise Exception(f"LLM request failed with status {response.status_code}")
                
        except Exception as e:
            # Fallback to context-based response
            print(f"[yellow]LLM not available ({e}), using context summary[/yellow]")
            
            # Simple context-based response
            if len(context_text) > 500:
                context_summary = context_text[:500] + "..."
            else:
                context_summary = context_text
            
            return f"Based on the available information: {context_summary}"
    
    def get_conversation_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of current conversation."""
        if self.chat_enabled and self.chat_history:
            return self.chat_history.get_conversation_summary()
        return None
    
    def clear_conversation(self):
        """Clear the current conversation history."""
        if self.chat_enabled and self.chat_history:
            self.chat_history.clear_history()
            print("[green]Conversation history cleared[/green]")
    
    def export_conversation(self, format: str = "json") -> Optional[str]:
        """Export current conversation history."""
        if self.chat_enabled and self.chat_history:
            return self.chat_history.export_history(format=format)
        return None

    def _reset_client(self):
        """Resets the ChromaDB client. Used for testing to release file locks."""
        if self.collection and self.collection._client:
            try:
                self.collection._client.reset()
            except Exception as e:
                print(f"[yellow]Warning: Failed to reset Chroma client: {e}[/yellow]")
