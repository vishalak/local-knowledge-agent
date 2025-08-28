#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm
from rich import print

import chromadb
from chromadb.utils import embedding_functions

STATE_DIR = ".kb_state"
STATE_FILE = "state.json"


def load_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Basic validation
    assert "source_dir" in cfg and cfg["source_dir"], "source_dir is required in config.yaml"
    cfg.setdefault("chroma_path", "./chroma_db")
    cfg.setdefault("collection", "local_kb")
    cfg.setdefault("include_extensions", [".md", ".txt", ".py"])
    cfg.setdefault("excludes", [".git", "node_modules", ".venv", "__pycache__", "build", "dist"])
    cfg.setdefault("chunk", {"size": 1200, "overlap": 200})
    cfg.setdefault("model", {"embedder": "BAAI/bge-small-en-v1.5"})
    return cfg


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def file_should_include(path: Path, include_exts: List[str], exclude_names: List[str]) -> bool:
    # Exclude directories by name anywhere in the path
    for part in path.parts:
        if part in exclude_names:
            return False
    # Only include files with configured extensions
    return path.suffix.lower() in set([e.lower() for e in include_exts])


def iter_files(root: Path, include_exts: List[str], exclude_names: List[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place for speed
        dirnames[:] = [d for d in dirnames if d not in exclude_names]
        for fn in filenames:
            p = Path(dirpath) / fn
            if file_should_include(p, include_exts, exclude_names):
                yield p


def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Splits text into overlapping character chunks.
    Returns list of tuples: (start_idx, end_idx, chunk_text)
    """
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
        start = max(0, end - overlap)
    return chunks


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"files": {}}
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def connect_collection(chroma_path: str, collection_name: str, embedder_model: str):
    client = chromadb.PersistentClient(path=chroma_path)
    st_embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedder_model)
    col = client.get_or_create_collection(name=collection_name, embedding_function=st_embed)
    return col


def delete_file_docs(collection, file_path: str):
    # Delete all chunks that belong to a file
    collection.delete(where={"path": file_path})


def upsert_file(collection, file_path: Path, chunk_size: int, overlap: int):
    b = file_path.read_bytes()
    text = None
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        # Try latin-1 fallback
        try:
            text = b.decode("latin-1")
        except Exception:
            print(f"[yellow]Skipping non-text file:[/yellow] {file_path}")
            return None

    chunks = chunk_text(text, chunk_size, overlap)
    file_hash = sha256_bytes(b)
    mtime = int(file_path.stat().st_mtime)

    ids = []
    docs = []
    metas = []

    # Use compact ids; Chroma requires unique string IDs
    from uuid import uuid4
    for (s, e, c) in chunks:
        ids.append(str(uuid4()))
        docs.append(c)
        metas.append({
            "path": str(file_path),
            "start": s,
            "end": e,
            "sha256": file_hash,
            "mtime": mtime,
        })

    if docs:
        # Batch upserts to avoid exceeding max batch size
        max_batch_size = 5000  # Conservative limit to avoid ChromaDB batch size errors
        for i in range(0, len(docs), max_batch_size):
            batch_ids = ids[i:i + max_batch_size]
            batch_docs = docs[i:i + max_batch_size]
            batch_metas = metas[i:i + max_batch_size]
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

    return {"sha256": file_hash, "mtime": mtime, "count": len(docs)}


def build(cfg_path: str):
    cfg = load_config(cfg_path)
    root = Path(cfg["source_dir"]).expanduser()
    chunk_size = int(cfg["chunk"]["size"])
    overlap = int(cfg["chunk"]["overlap"])

    collection = connect_collection(cfg["chroma_path"], cfg["collection"], cfg["model"]["embedder"])

    state_path = Path(STATE_DIR) / STATE_FILE
    # Fresh build: wipe old collection by deleting and recreating
    print("[cyan]Starting fresh build...[/cyan]")
    # Dangerous: collection.reset() resets *all* collections for client; avoid.
    # Instead, delete all docs matching this collection by dropping and recreating.
    # There is no drop in Chroma client, so we delete by no filter -> deletes entire collection.
    collection.delete()  # delete everything in this collection

    state = {"files": {}}

    files = list(iter_files(root, cfg["include_extensions"], cfg["excludes"]))
    for p in tqdm(files, desc="Indexing", unit="file"):
        info = upsert_file(collection, p, chunk_size, overlap)
        if info:
            state["files"][str(p)] = info

    save_state(state_path, state)
    print(f"[green]Build complete.[/green] Indexed {len(state['files'])} files.")


def update(cfg_path: str):
    cfg = load_config(cfg_path)
    root = Path(cfg["source_dir"]).expanduser()
    chunk_size = int(cfg["chunk"]["size"])
    overlap = int(cfg["chunk"]["overlap"])

    collection = connect_collection(cfg["chroma_path"], cfg["collection"], cfg["model"]["embedder"])
    state_path = Path(STATE_DIR) / STATE_FILE
    state = load_state(state_path)

    current_files = set()
    added, changed, unchanged = 0, 0, 0

    for p in tqdm(list(iter_files(root, cfg["include_extensions"], cfg["excludes"])), desc="Scanning", unit="file"):
        current_files.add(str(p))
        b = p.read_bytes()
        sha = sha256_bytes(b)

        prev = state.get("files", {}).get(str(p))
        if prev and prev.get("sha256") == sha:
            unchanged += 1
            continue

        # New or changed
        delete_file_docs(collection, str(p))  # clear old chunks
        info = upsert_file(collection, p, chunk_size, overlap)
        if info:
            state["files"][str(p)] = info
            if prev is None:
                added += 1
            else:
                changed += 1

    # Removed files
    removed_files = [fp for fp in list(state.get("files", {}).keys()) if fp not in current_files]
    for fp in removed_files:
        delete_file_docs(collection, fp)
        del state["files"][fp]

    save_state(state_path, state)

    print(f"[green]Update complete.[/green] Files - added: {added}, changed: {changed}, removed: {len(removed_files)}, unchanged: {unchanged}.")


def rebuild(cfg_path: str):
    # Alias for full rebuild
    build(cfg_path)


def main():
    parser = argparse.ArgumentParser(description="Local KB indexer (Chroma)")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Fresh build of the vector DB")
    sub.add_parser("update", help="Incremental update (add/change/remove files)")
    sub.add_parser("rebuild", help="Full rebuild (same as build)")

    args = parser.parse_args()

    if args.cmd == "build":
        build(args.config)
    elif args.cmd == "update":
        update(args.config)
    elif args.cmd == "rebuild":
        rebuild(args.config)
    else:
        raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
