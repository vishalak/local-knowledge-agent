# 🧠 Local Knowledge Base Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **powerful, private, and fully local** Retrieval-Augmented Generation (RAG) system that transforms your codebase and documentation into an intelligent, queryable knowledge base. No data ever leaves your machine!

## ✨ Key Features

### 🔒 **Complete Privacy & Offline Operation**

- **100% Local Processing** - All embeddings, storage, and generation happen on your machine
- **No Data Upload** - Your code and documents never leave your environment
- **Works Offline** - Once models are downloaded, runs completely offline

### 🚀 **Advanced Document Processing**

- **Multi-Format Support** - Code files, PDFs, Word documents, Markdown, and more
- **Language-Aware Processing** - Smart chunking for Python, JavaScript, C#, and other languages
- **Semantic Chunking** - Groups related content intelligently for better context
- **Parallel Processing** - Fast indexing with concurrent file processing
- **Incremental Updates** - Only processes changed files for efficiency

### 🎯 **Intelligent Retrieval & Generation**

- **HyDE Query Enhancement** - Transforms queries for better search results
- **Multi-Strategy Reranking** - BM25, semantic, and hybrid approaches
- **Conversational Context** - Maintains chat history for natural interactions
- **Streaming Responses** - Real-time generation with live feedback

### 💻 **Multiple Interfaces**

- **CLI Tool** - Full-featured command-line interface with rich formatting
- **Web UI** - Modern Streamlit-based web interface with chat functionality
- **Enhanced Citations** - Detailed source attribution with VS Code integration

### 🛠 **Developer-Focused**

- **VS Code Integration** - Direct links to source files at specific lines
- **Rich Metadata** - File types, modification dates, sizes, and content previews
- **Code-Aware** - Understands namespaces, classes, functions, and methods
- **Comprehensive Testing** - 46+ unit and integration tests for reliability

---

## 🚀 Quick Start

### 1. Prerequisites

**Install Required Software:**

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Ollama** - [Download](https://ollama.com/) for local LLM

**Setup Ollama:**

```bash
# Download and install Ollama, then pull a model
ollama pull llama3:8b
# Test that it works
ollama run llama3:8b
```

### 2. Installation

```bash
# Clone or download the project
cd local_kb_agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.yaml` to point to your codebase:

```yaml
# === Local Knowledge Assistant - Configuration ===
source_dir: "C:/your-project-folder"  # 👈 Change this to your code directory

# File types to include (supports many formats)
include_extensions:
  - ".md"
  - ".txt" 
  - ".py"
  - ".js"
  - ".jsx"
  - ".ts"
  - ".tsx"
  - ".cs"         # C# support
  - ".java"
  - ".pdf"        # PDF documents
  - ".docx"       # Word documents
  # ... and more

# Advanced settings (optional)
chunk:
  size: 1200              # Chunk size for processing
  overlap: 200            # Overlap between chunks
  semantic: true          # Enable semantic chunking

retrieval:
  query_transform: "hyde" # Query enhancement method
  rerank_method: "bm25"   # Reranking strategy
  max_results: 10         # Results to consider

chat:
  enabled: true           # Conversational context
  max_history: 10         # Chat history length
```

### 4. Build Knowledge Base

**Initial build (first time):**

```bash
python src/index_kb.py
```

**Update after changes:**

```bash
python src/index_kb.py --update
```

**Full rebuild:**

```bash
python src/index_kb.py --rebuild
```

### 5. Start Querying

**Command Line Interface:**

```bash
# Basic query
python src/ask.py "How do I configure the database connection?"

# With enhanced citations
python src/ask.py "Show me authentication code examples" --enhanced-citations

# With VS Code links
python src/ask.py "Find error handling patterns" --vscode-links

# Interactive mode with streaming
python src/ask.py --interactive --stream
```

**Web Interface:**

```bash
# Launch the web UI
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

---

## 🎮 Usage Examples

### CLI Examples

```bash
# Basic usage
python src/ask.py "How is user authentication implemented?"

# Show detailed sources with metadata
python src/ask.py "Find API endpoint definitions" --enhanced-citations

# Interactive chat mode with streaming
python src/ask.py --interactive --stream

# Generate VS Code links for easy navigation
python src/ask.py "Show database schema code" --vscode-links

# Adjust retrieval settings
python src/ask.py "Find configuration examples" --k 10

# Clear conversation history
python src/ask.py --clear-history
```

### Interactive Commands

In interactive mode, use these commands:

- `/clear` - Clear conversation history
- `/history` - Show conversation summary  
- `/sources` - Toggle source display
- `/enhanced` - Toggle enhanced citations
- `/inline` - Toggle inline citations
- `/vscode` - Toggle VS Code links
- `/stream` - Toggle streaming responses
- `quit` or `exit` - Exit interactive mode

---

## 🌐 Web Interface Features

The Streamlit web UI provides a modern, user-friendly interface with:

### 🎨 **Modern Design**

- Clean, professional chat interface
- Responsive design for desktop and mobile
- Dark/light theme support
- Customizable settings sidebar

### 💬 **Chat Experience**

- Real-time streaming responses
- Persistent conversation history
- Message threading and context
- Visual feedback and progress indicators

### 📚 **Enhanced Source Display**

- Expandable source citations with metadata
- Content previews and file information
- Direct VS Code integration links
- Relevance scoring and ranking

### ⚙️ **Configurable Settings**

- Adjustable retrieval parameters (k-value)
- Toggle streaming responses
- Control source display options
- Knowledge base statistics

---

## 🏗️ Architecture

### Core Components

```html
local_kb_agent/
├── src/
│   ├── kb.py                    # 🧠 Main KnowledgeBase class
│   ├── ask.py                   # 💬 CLI interface
│   ├── index_kb.py              # 📥 Indexing functionality
│   ├── document_processors.py   # 📄 Multi-format text extraction
│   ├── code_splitters.py        # 🔧 Language-aware chunking
│   ├── semantic_chunker.py      # 🎯 Semantic grouping
│   ├── metadata_extractor.py    # 📊 File metadata extraction
│   ├── query_transformer.py     # 🔍 Query enhancement (HyDE)
│   ├── reranker.py             # 📈 Result reranking
│   ├── chat_history.py         # 💭 Conversation management
│   └── source_citation.py      # 📝 Enhanced citations
├── streamlit_app.py            # 🌐 Web UI
├── tests/                      # 🧪 Comprehensive test suite
├── docs/                       # 📚 Documentation
└── config.yaml                # ⚙️ Configuration
```

### Data Flow

1. **📥 Document Ingestion**
   - File discovery and filtering
   - Multi-format text extraction (PDF, Word, code)
   - Language-aware chunking
   - Metadata extraction

2. **🔍 Embedding & Storage**
   - Local sentence transformer embeddings
   - ChromaDB vector storage
   - Incremental updates and deduplication

3. **🎯 Query Processing**
   - HyDE query transformation
   - Vector similarity search
   - Multi-strategy reranking (BM25, semantic, hybrid)

4. **🤖 Response Generation**
   - Context preparation and injection
   - Local LLM generation via Ollama
   - Streaming response delivery
   - Enhanced source citation

---

## 🔧 Advanced Configuration

### Model Configuration

```yaml
model:
  embedder: "BAAI/bge-small-en-v1.5"  # Embedding model
  llm: "llama3:8b"                    # Ollama LLM model

# Alternative models:
# embedder: "BAAI/bge-base-en-v1.5"     # Better quality, more memory
# embedder: "jinaai/jina-embeddings-v3"  # Code-specialized
# llm: "llama3:70b"                      # Larger, better quality
# llm: "mistral"                         # Alternative model
```

### Chunking Strategies

```yaml
chunk:
  size: 1200        # Base chunk size
  overlap: 200      # Overlap between chunks
  semantic: true    # Enable semantic chunking

# For different content types:
# - Code: smaller chunks (800-1000) for functions/classes
# - Documentation: larger chunks (1500-2000) for coherent sections
# - Mixed: default (1200) works well
```

### Retrieval Tuning

```yaml
retrieval:
  query_transform: "hyde"     # "hyde", "simple", or "none"
  max_results: 10             # Initial retrieval count
  rerank_method: "hybrid"     # "bm25", "semantic", "hybrid", "none"
  rerank_max_results: 5       # Final result count
  vector_weight: 0.7          # Hybrid: vector similarity weight
  bm25_weight: 0.3           # Hybrid: BM25 weight
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_kb.py -v           # Core functionality
python -m pytest tests/test_indexing.py -v     # Indexing features
python -m pytest tests/test_retrieval.py -v    # Search and retrieval

# Generate coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

Current test coverage: **46+ tests** covering all major functionality.

---

## 📊 Performance Guidelines

### Hardware Requirements

**Minimum:**

- 8GB RAM
- 4GB free disk space
- Python 3.10+

**Recommended:**

- 16GB+ RAM (for large codebases)
- SSD storage (faster indexing)
- GPU (optional, for faster embeddings)

### Optimization Tips

**For Large Codebases (>10,000 files):**

```yaml
# Increase batch size for faster processing
chunk:
  size: 1000
  overlap: 150

# Use more efficient embedding model
model:
  embedder: "BAAI/bge-small-en-v1.5"

# Optimize exclusions
excludes:
  - ".git"
  - "node_modules" 
  - "__pycache__"
  - "build"
  - "dist"
  - "*.log"
```

**For Better Quality:**

```yaml
# Larger, better embedding model
model:
  embedder: "BAAI/bge-base-en-v1.5"

# More sophisticated reranking
retrieval:
  rerank_method: "hybrid"
  max_results: 15
  rerank_max_results: 8
```

---

## 🔌 Integration & Extensions

### VS Code Integration

The system generates `vscode://` links for direct file access:

```bash
# Enable VS Code links in CLI
python src/ask.py "find authentication code" --vscode-links

# In web UI: click the "Open in VS Code" buttons in source citations
```

### CI/CD Integration

Add knowledge base health checks to your CI pipeline:

```yaml
# .github/workflows/kb-health.yml
name: Knowledge Base Health Check
on: [push, pull_request]
jobs:
  test-kb:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Test knowledge base
        run: python -m pytest tests/
```

### API Mode

Use the KnowledgeBase class in your own applications:

```python
from src.kb import KnowledgeBase

# Initialize
kb = KnowledgeBase("config.yaml")

# Query programmatically
result = kb.ask_with_context("How is authentication handled?")
print(result['response'])

# Access sources
for source in result['context_docs']:
    print(f"Source: {source['path']}:{source['start']}-{source['end']}")
```

---

## 🛠️ Troubleshooting

### Common Issues

**"No results in local knowledge base"**

```bash
# Rebuild the knowledge base
python src/index_kb.py --rebuild

# Check configuration
python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['source_dir'])"
```

**Ollama connection errors**

```bash
# Check if Ollama is running
ollama list

# Test model availability
ollama run llama3:8b "Hello"

# Check different port (if changed)
export OLLAMA_HOST=http://localhost:11435
```

**Memory issues with large codebases**

```yaml
# Reduce chunk size and batch processing
chunk:
  size: 800
  overlap: 100

# Use smaller embedding model
model:
  embedder: "BAAI/bge-small-en-v1.5"
```

**Unicode/encoding errors**

```yaml
# Add file encoding hints
excludes:
  - "*.bin"
  - "*.exe"
  - "*/.git/*"
```

### Debugging

Enable verbose output:

```bash
# Verbose indexing
python src/index_kb.py --verbose

# Debug query processing
python src/ask.py "test query" --show-sources --enhanced-citations
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run the test suite**: `python -m pytest tests/`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install black pytest pytest-cov

# Run code formatting
black src/ tests/

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## 🙏 Acknowledgments

- **ChromaDB** - Vector database foundation
- **Sentence Transformers** - Embedding models
- **Ollama** - Local LLM infrastructure
- **Streamlit** - Web UI framework
- **Rich** - Terminal formatting
- **PyPDF2** - PDF processing
- **python-docx** - Word document processing

---

## 📞 Support

- **Documentation**: [docs/README.md](docs/README.md)
- **Issues**: [GitHub Issues](https://github.com/your-repo/local_kb_agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/local_kb_agent/discussions)

---

## **Happy Knowledge Hunting! 🎯**

*Transform your codebase into an intelligent, searchable knowledge base that works entirely on your machine.*
