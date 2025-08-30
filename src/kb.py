#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
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

STATE_DIR = ".kb_state"
STATE_FILE = "state.json"

class KnowledgeBase:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.cfg = self._load_config()
        self.collection = self._connect_collection()
        self.state_path = Path(STATE_DIR) / STATE_FILE
        self.state = self._load_state()

    def _load_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Basic validation
        assert "source_dir" in cfg and cfg["source_dir"], "source_dir is required in config.yaml"
        cfg.setdefault("chroma_path", "./chroma_db")
        cfg.setdefault("collection", "local_kb")
        cfg.setdefault("include_extensions", [".md", ".txt", ".py", ".pdf", ".docx", ".doc"])
        cfg.setdefault("excludes", [".git", "node_modules", ".venv", "__pycache__", "build", "dist"])
        cfg.setdefault("chunk", {"size": 1200, "overlap": 200, "semantic": True})
        cfg.setdefault("metadata", {"generate_summaries": False, "extract_advanced": True})
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
        res = self.collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
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

    def _reset_client(self):
        """Resets the ChromaDB client. Used for testing to release file locks."""
        if self.collection and self.collection._client:
            try:
                self.collection._client.reset()
            except Exception as e:
                print(f"[yellow]Warning: Failed to reset Chroma client: {e}[/yellow]")
