# 🔧 Developer Documentation

This document provides detailed technical information for developers working with the Local Knowledge Base Agent.

## 📋 Table of Contents

- [Architecture Overview](#️-architecture-overview)
- [Core Components](#-core-components)
- [API Reference](#-api-reference)
- [Testing Guide](#-testing-guide)
- [Contributing Guidelines](#-contributing-guidelines)
- [Performance Optimization](#-performance-optimization)

---

## 🏗️ Architecture Overview

### System Design

The Local Knowledge Base Agent follows a modular architecture with clear separation of concerns:

```html
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Layer    │    │  Interface      │    │   Core Engine   │
│                 │    │                 │    │                 │
│ • CLI (ask.py)  │◄──►│ • KnowledgeBase │◄──►│ • Document Proc │
│ • Web UI        │    │   Class         │    │ • Embeddings    │
│ • API           │    │ • Query Handler │    │ • Vector Store  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   AI/ML Layer   │    │  Storage Layer  │
│                 │    │                 │    │                 │
│ • File System   │    │ • Ollama LLM    │    │ • ChromaDB      │
│ • Documents     │    │ • Transformers  │    │ • State Files   │
│ • Metadata      │    │ • Reranking     │    │ • Chat History  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Document Ingestion Pipeline**

   ```html
   Files → Document Processors → Code Splitters → Semantic Chunker → Metadata Extractor → ChromaDB
   ```

2. **Query Processing Pipeline**

   ```html
   User Query → Query Transformer → Vector Search → Reranker → Context Builder → LLM → Response
   ```

3. **Response Enhancement Pipeline**

   ```html
   LLM Response → Citation Manager → Chat History → UI Formatter → User
   ```

---

## 🧩 Core Components

### 1. KnowledgeBase Class (`src/kb.py`)

The central orchestrator that manages all operations:

```python
class KnowledgeBase:
    def __init__(self, config_path: str = "config.yaml")
    def index_documents(self, force_rebuild: bool = False)
    def ask_with_context(self, query: str, k: int = 6, stream: bool = False)
    def retrieve_context(self, query: str, k: int = 6)
    def clear_conversation(self)
```

**Key Methods:**

- `index_documents()`: Processes and indexes all documents
- `ask_with_context()`: Main query interface with optional streaming
- `retrieve_context()`: Gets relevant documents for a query
- `_generate_response()`: Handles LLM interaction

### 2. Document Processors (`src/document_processors.py`)

Handles multi-format text extraction:

```python
def extract_text_from_file(file_path: Path) -> str
def extract_pdf_text(file_path: Path) -> str
def extract_docx_text(file_path: Path) -> str
def extract_code_text(file_path: Path) -> str
```

**Supported Formats:**

- Code files (Python, JavaScript, C#, Java, etc.)
- PDF documents (via PyPDF2)
- Word documents (via python-docx)
- Plain text and Markdown

### 3. Code Splitters (`src/code_splitters.py`)

Language-aware chunking for better code understanding:

```python
def get_code_aware_chunks(text: str, file_path: str, chunk_size: int, overlap: int)
def split_python_code(text: str, chunk_size: int, overlap: int)
def split_javascript_code(text: str, chunk_size: int, overlap: int)
def split_csharp_code(text: str, chunk_size: int, overlap: int)
```

**Features:**

- Respects function and class boundaries
- Preserves import statements
- Handles nested structures
- Language-specific syntax awareness

### 4. Semantic Chunker (`src/semantic_chunker.py`)

Groups semantically related content:

```python
def get_semantic_chunks(text: str, max_chunk_size: int, similarity_threshold: float)
def calculate_sentence_similarities(sentences: List[str])
def group_similar_sentences(sentences: List[str], similarities: np.ndarray)
```

**Algorithm:**

1. Split text into sentences
2. Calculate semantic similarities
3. Group adjacent similar sentences
4. Respect maximum chunk size limits

### 5. Query Transformer (`src/query_transformer.py`)

Enhances queries for better retrieval:

```python
def transform_query(query: str, method: str = "hyde")
def hyde_transform(query: str)
def simple_transform(query: str)
```

**HyDE (Hypothetical Document Embeddings):**

- Generates hypothetical answer to the query
- Uses this synthetic content for better vector search
- Significantly improves retrieval quality

### 6. Reranker (`src/reranker.py`)

Improves result precision through reranking:

```python
class BM25Reranker:
    def rerank_documents(self, query: str, documents: List[Dict], max_results: int)

class SemanticReranker:
    def rerank_documents(self, query: str, documents: List[Dict], max_results: int)

class HybridReranker:
    def rerank_documents(self, query: str, documents: List[Dict], max_results: int)
```

**Strategies:**

- **BM25**: Term frequency-based ranking
- **Semantic**: Embedding similarity-based
- **Hybrid**: Combines both approaches with configurable weights

### 7. Source Citation Manager (`src/source_citation.py`)

Enhanced source attribution and formatting:

```python
class SourceCitationManager:
    def format_sources_for_display(self, context_docs: List[Dict], format_type: str)
    def enhance_response_with_citations(self, response: str, context_docs: List[Dict])
    def generate_vscode_links(self, context_docs: List[Dict])
```

**Features:**

- Rich metadata display
- Content previews
- VS Code integration links
- Multiple output formats (terminal, web, markdown)

---

## 📚 API Reference

### Configuration Schema

```yaml
# config.yaml structure
source_dir: string              # Required: source directory path
chroma_path: string            # ChromaDB storage path
collection: string             # Collection name
include_extensions: list       # File extensions to include
excludes: list                 # Paths/patterns to exclude

chunk:
  size: int                    # Base chunk size (default: 1200)
  overlap: int                 # Chunk overlap (default: 200)
  semantic: bool               # Enable semantic chunking (default: true)

metadata:
  generate_summaries: bool     # Generate content summaries
  extract_advanced: bool       # Extract detailed metadata

retrieval:
  query_transform: string      # "hyde", "simple", or "none"
  max_results: int            # Initial retrieval count
  rerank_method: string       # "bm25", "semantic", "hybrid", "none"
  rerank_max_results: int     # Final result count
  vector_weight: float        # Hybrid reranking vector weight
  bm25_weight: float         # Hybrid reranking BM25 weight

chat:
  enabled: bool               # Enable conversation history
  max_history: int           # Maximum exchanges to remember
  use_context: bool          # Use history for query enhancement
  max_context_exchanges: int  # History items for context

model:
  embedder: string           # Sentence transformer model
  llm: string               # Ollama model name
```

### KnowledgeBase API

```python
# Initialize
kb = KnowledgeBase("config.yaml")

# Index documents
kb.index_documents(force_rebuild=False)

# Query with context
result = kb.ask_with_context(
    query="How is authentication handled?",
    k=6,                    # Number of sources to retrieve
    stream=False           # Enable streaming responses
)

# Result structure
{
    'response': str,           # Generated answer
    'context_docs': [          # Source documents
        {
            'text': str,       # Document content
            'path': str,       # File path
            'start': int,      # Start line/position
            'end': int,        # End line/position
            'distance': float, # Vector similarity distance
            'rank': int,       # Result ranking
            # ... additional metadata
        }
    ],
    'enhanced_query': str,     # Transformed query
    'response_time': float,    # Processing time
    'num_context_docs': int   # Number of sources used
}

# Streaming response
if stream=True:
    result['response_stream'] = Iterator[str]  # Streaming generator
```

### CLI API

```bash
# Basic usage
python src/ask.py "query text" [options]

# Options
--config, -c PATH          # Config file path
--k INT                    # Number of sources (default: 6)
--model, -m MODEL         # Override Ollama model
--show-sources            # Display basic source info
--enhanced-citations      # Rich source citations with metadata
--inline-citations        # Include citations in response text
--vscode-links           # Generate VS Code file links
--stream                 # Enable streaming responses
--interactive, -i        # Interactive chat mode
--clear-history          # Clear conversation history
--show-history           # Display conversation summary

# Interactive commands
/clear                   # Clear conversation history
/history                 # Show conversation summary
/sources                 # Toggle source display
/enhanced                # Toggle enhanced citations
/inline                  # Toggle inline citations
/vscode                  # Toggle VS Code links
/stream                  # Toggle streaming mode
quit, exit               # Exit interactive mode
```

### Web UI API

The Streamlit web interface provides a programmatic way to interact:

```python
# Custom Streamlit components
@st.cache_resource
def load_knowledge_base(config_path: str) -> KnowledgeBase

def format_source_for_display(source: Dict, index: int) -> Dict
def display_sources(sources: List[Dict])
```

---

## 🧪 Testing Guide

### Test Structure

```html
tests/
├── __init__.py
├── test_kb.py                    # Core KnowledgeBase functionality
├── test_indexing.py             # Document indexing and processing
├── test_retrieval.py            # Search and retrieval features
├── test_document_processors.py  # File format processing
├── test_code_splitters.py       # Language-aware chunking
├── test_semantic_chunker.py     # Semantic grouping
├── test_metadata_extractor.py   # Metadata extraction
├── test_query_transformer.py    # Query enhancement
├── test_reranker.py             # Result reranking
├── test_chat_history.py         # Conversation management
├── test_source_citation.py      # Citation formatting
└── conftest.py                  # Test configuration and fixtures
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_kb.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test function
python -m pytest tests/test_kb.py::test_ask_with_context -v

# Run tests matching pattern
python -m pytest tests/ -k "test_retrieval" -v
```

### Test Categories

**Unit Tests:**

- Individual function testing
- Mocked dependencies
- Fast execution
- High coverage

**Integration Tests:**

- End-to-end workflows
- Real file processing
- ChromaDB interactions
- Ollama integration

**Performance Tests:**

- Indexing speed benchmarks
- Query response times
- Memory usage monitoring
- Large dataset handling

### Writing Tests

```python
# Example test structure
import pytest
from src.kb import KnowledgeBase

class TestKnowledgeBase:
    def setup_method(self):
        """Setup before each test"""
        self.kb = KnowledgeBase("test_config.yaml")
    
    def test_indexing(self):
        """Test document indexing"""
        result = self.kb.index_documents()
        assert result['documents_processed'] > 0
        assert result['chunks_created'] > 0
    
    def test_query_processing(self):
        """Test query handling"""
        result = self.kb.ask_with_context("test query")
        assert 'response' in result
        assert 'context_docs' in result
        assert len(result['context_docs']) > 0
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test with large dataset (marked as slow)"""
        # Integration test code
        pass
```

### Test Configuration

```yaml
# test_config.yaml
source_dir: "test_data"
chroma_path: "./test_chroma_db"
collection: "test_collection"
include_extensions: [".py", ".md", ".txt"]
excludes: ["__pycache__", ".git"]

chunk:
  size: 500
  overlap: 50
  semantic: false

model:
  embedder: "BAAI/bge-small-en-v1.5"
  llm: "llama3:8b"
```

---

## 🤝 Contributing Guidelines

### Development Workflow

1. **Fork and Clone**

   ```bash
   git clone https://github.com/your-fork/local_kb_agent.git
   cd local_kb_agent
   ```

2. **Setup Development Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install black pytest pytest-cov
   ```

3. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Follow existing code style
   - Add comprehensive tests
   - Update documentation

5. **Run Quality Checks**

   ```bash
   # Format code
   black src/ tests/
   
   # Run tests
   python -m pytest tests/ --cov=src
   
   # Lint if desired
   flake8 src/ tests/
   ```

6. **Commit and Push**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**

### Code Style

**Formatting:**

- Use Black for code formatting
- Line length: 88 characters
- Use type hints where possible

**Naming Conventions:**

- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

**Documentation:**

- Docstrings for all public functions
- Type hints for function signatures
- Inline comments for complex logic

### Commit Messages

Follow conventional commits format:

```html
feat: add semantic chunking support
fix: resolve unicode handling in PDF processor
docs: update API documentation
test: add integration tests for reranking
refactor: extract citation formatting logic
perf: optimize vector search performance
```

### Pull Request Guidelines

**PR Requirements:**

- Clear description of changes
- Tests for new functionality
- Documentation updates
- No breaking changes without discussion
- Passes all CI checks

**PR Template:**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
```

---

## ⚡ Performance Optimization

### Indexing Performance

**Batch Processing:**

```python
# Optimize batch sizes for your hardware
EMBEDDING_BATCH_SIZE = 100    # Adjust based on GPU memory
PROCESSING_BATCH_SIZE = 50    # File processing concurrency
```

**Parallel Processing:**

```python
# Configure thread pools
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Process files in parallel
```

**Memory Management:**

```python
# Chunk large files to prevent memory issues
def process_large_file(file_path, max_chunk_size=10_000_000):
    for chunk in read_file_chunks(file_path, max_chunk_size):
        yield process_chunk(chunk)
```

### Query Performance

**Embedding Caching:**

```python
# Cache embeddings for repeated queries
@lru_cache(maxsize=1000)
def get_query_embedding(query: str) -> np.ndarray:
    return embedding_model.encode(query)
```

**Result Caching:**

```python
# Cache search results for identical queries
@lru_cache(maxsize=500)
def cached_vector_search(query_hash: str, k: int) -> List[Dict]:
    return vector_search(query_hash, k)
```

**Database Optimization:**

```python
# ChromaDB performance settings
client = chromadb.PersistentClient(
    path=chroma_path,
    settings=chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_path,
        anonymized_telemetry=False
    )
)
```

### Memory Optimization

**Streaming Processing:**

```python
def stream_documents():
    """Process documents in streaming fashion to reduce memory usage"""
    for file_path in get_file_paths():
        with open(file_path, 'r') as f:
            for chunk in read_chunks(f):
                yield process_chunk(chunk)
```

**Garbage Collection:**

```python
import gc

def cleanup_after_processing():
    """Force garbage collection after heavy operations"""
    gc.collect()
    torch.cuda.empty_cache()  # If using GPU
```

### Monitoring

**Performance Metrics:**

```python
import time
import psutil
from typing import Dict

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'peak_memory': psutil.Process().memory_info().rss
        }
        
        print(f"Performance: {func.__name__} - {metrics}")
        return result
    return wrapper
```

**Usage Analytics:**

```python
def log_query_metrics(query: str, response_time: float, num_results: int):
    """Log query performance for analysis"""
    metrics = {
        'timestamp': time.time(),
        'query_length': len(query),
        'response_time': response_time,
        'num_results': num_results
    }
    # Log to file or monitoring system
```

---

## 🔧 Advanced Configuration

### Custom Embedding Models

```python
# Add custom embedding model support
class CustomEmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode([text], convert_to_numpy=True)[0]
```

### Custom Document Processors

```python
# Add support for new file formats
def extract_custom_format(file_path: Path) -> str:
    """Extract text from custom file format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Custom processing logic
        return processed_content

# Register processor
DOCUMENT_PROCESSORS['.custom'] = extract_custom_format
```

### Custom Reranking Strategies

```python
class CustomReranker:
    def __init__(self, config: Dict):
        self.config = config
    
    def rerank_documents(self, query: str, documents: List[Dict], 
                        max_results: int) -> List[Dict]:
        """Custom reranking logic"""
        # Implement your reranking strategy
        return reranked_documents[:max_results]
```

---

## **Happy Developing! 🚀**

This documentation should provide everything you need to understand, use, and extend the Local Knowledge Base Agent. For additional help, check the main [README.md](../README.md) or open an issue on GitHub.
