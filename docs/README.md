# Local Knowledge Assistant (Private, On-Device)

This kit lets you run a **local RAG (Retrieval-Augmented Generation)** assistant that:

- Indexes your private code and docs on your machine (no uploads).
- Stores embeddings in a local **Chroma** vector DB.
- Answers questions using a local LLM via **Ollama**.
- Supports **incremental updates** when you add/modify/remove files.

> Works on Windows, macOS, and Linux. The instructions below are Windows-friendly.

---

## 1) Install prerequisites

1. **Python 3.10+**
2. **Ollama** (for a local LLM)
   - Download: <https://ollama.com/>
   - After install, pull a model (example):

     ```bash
     ollama run llama3:8b
     ```

     (This first run downloads the model.)

3. Create and activate a virtual environment, then install Python deps:

   ```bash
   cd local_kb_agent
   python -m venv .venv
   .venv\Scripts\activate  # Windows (PowerShell: .venv\Scripts\Activate.ps1)
   pip install -r requirements.txt
   ```

---

## 2) Configure

Edit `config.yaml`:

- Set `source_dir` to your **code_and_docs** root folder.
- Optionally adjust `include_extensions`, `excludes`, and chunk size/overlap.
- Change `model.llm` to another Ollama model if desired (e.g., `qwen2.5:7b`, `mistral` etc.).

---

## 3) Build the local vector DB

**Fresh build** (first time or full rebuild):

```bash
python index_kb.py build
```

This scans `source_dir`, chunks files, embeds them locally, and writes to the Chroma DB at `chroma_path`.
It also writes a small state file at `.kb_state/state.json` to track file hashes.

---

## 4) Ask questions

Make sure Ollama is running (it runs a local server in the background), then:

```bash
python ask.py "How do I configure the S3 client in our project?"
```

Show sources used:

```bash
python ask.py --show-sources "Where is the GitHub Actions workflow that builds the Docker image?"
```

Change number of retrieved chunks (top-k):

```bash
python ask.py -k 8 "What does the custom retry middleware do?"
```

Override model at runtime:

```bash
python ask.py -m "llama3:8b-instruct" "Explain the startup sequence"
```

---

## 5) Update when you add/modify files

When you add new folders or files under `source_dir`, or change/delete existing files, run:

```bash
python index_kb.py update
```

- **Added files** → embedded and inserted
- **Changed files** → previous chunks removed and re-indexed
- **Removed files** → their chunks deleted from the DB

You can run `update` as often as you like; it’s idempotent and efficient.

If you want a **full rebuild**, run:

```bash
python index_kb.py rebuild
```

(which is the same as `build` but keeps the command semantics explicit).

---

## 6) Tips & options

- **Performance**: If your corpus is big, switch to `BAAI/bge-base-en-v1.5` (larger, better) or keep `bge-small` (fast, RAM-light).
- **Chunking**: Increase `chunk.size` for long files; keep `overlap` ~15–25% of size.
- **Citations**: The CLI prints which files/byte ranges it used with `--show-sources`.
- **Ignored paths**: Tune `excludes` for speed (`node_modules`, `.git`, build artifacts, etc.).
- **Security**: Your data never leaves your machine. Embeddings are computed locally.

---

## 7) (Optional) Next steps

- Add a simple **Slack/Teams bot** that calls `ask.py` under the hood.
- Add **web-docs fallback** for Git/GitHub (live retrieval) guarded by a domain allowlist.
- Swap in a **code-aware embedding model** like `jinaai/jina-embeddings-v3-base-code` for better code search.
- For PDFs, add a PDF loader (e.g., `pypdf`) and extract text before chunking.

---

## Troubleshooting

- **"No results in local knowledge base."**  
  Run `python index_kb.py build` first, and confirm `source_dir` is correct.

- **Ollama errors or long responses**  
  Ensure `ollama run llama3:8b` works from the terminal. Try a smaller model if memory is tight.

- **UnicodeDecodeError**  
  The indexer falls back to latin-1. You can customize encodings per file type if needed.

---

Happy indexing! 🎯
