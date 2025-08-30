# Project Improvement Plan

This document outlines the plan for enhancing the Local Knowledge Base Agent project.

## To-Do Checklist

### 1. Improve Code Quality and Maintainability

- [x] Refactor `index_kb.py` and `ask.py` into a `KnowledgeBase` class.
- [x] Encapsulate configuration, database connection, and state management within the class.
- [x] Simplify `index_kb.py` and `ask.py` to act as entry points using the new class.
- [x] Add unit and integration tests for core logic (chunking, indexing, querying).

### 2. Enhance Document Processing and Indexing

- [ ] Add support for more file types (PDFs, Word documents).
- [ ] Implement code-aware splitting for source code files.
- [ ] Use parallel processing to speed up indexing.
- [ ] Explore semantic chunking for more meaningful text splits.
- [ ] Extract and store more advanced metadata (e.g., creation dates, summaries).

### 3. Improve Retrieval and Generation Quality

- [ ] Implement query transformations (e.g., HyDE) to refine user queries.
- [ ] Add a reranking step to improve the precision of retrieved documents.
- [ ] Incorporate chat history to make the agent conversational.

### 4. Enhance User Experience and Interface

- [ ] Implement streaming for LLM responses to show real-time generation.
- [ ] Cite sources for each answer to build trust and allow verification.
- [ ] Build a simple web UI using Streamlit or Gradio for easier interaction.
