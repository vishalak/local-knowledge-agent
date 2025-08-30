import unittest
from unittest.mock import Mock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reranker import DocumentReranker, create_reranker


class TestDocumentReranker(unittest.TestCase):
    
    def setUp(self):
        """Set up test documents."""
        self.query = "python machine learning"
        self.documents = [
            {
                "content": "Python is a programming language used for machine learning and data science.",
                "path": "/doc1.txt",
                "distance": 0.2
            },
            {
                "content": "Java is a popular programming language for enterprise applications.",
                "path": "/doc2.txt", 
                "distance": 0.8
            },
            {
                "content": "Machine learning algorithms can be implemented in Python using libraries like scikit-learn.",
                "path": "/doc3.txt",
                "distance": 0.3
            },
            {
                "content": "Web development with JavaScript and Node.js for modern applications.",
                "path": "/doc4.txt",
                "distance": 0.9
            }
        ]
    
    def test_no_reranking(self):
        """Test that 'none' method returns documents unchanged."""
        reranker = DocumentReranker(method="none")
        result = reranker.rerank_documents(self.query, self.documents)
        
        self.assertEqual(len(result), len(self.documents))
        self.assertEqual(result, self.documents)
    
    def test_bm25_reranking(self):
        """Test BM25 reranking functionality."""
        reranker = DocumentReranker(method="bm25")
        result = reranker.rerank_documents(self.query, self.documents, max_results=3)
        
        self.assertLessEqual(len(result), 3)
        
        # Check that BM25 scores are added
        for doc in result:
            self.assertIn("bm25_score", doc)
            self.assertIsInstance(doc["bm25_score"], (int, float))
        
        # Documents with "python" and "machine learning" should rank higher
        relevant_docs = [doc for doc in result if "python" in doc["content"].lower() 
                        and "machine" in doc["content"].lower()]
        self.assertGreater(len(relevant_docs), 0)
    
    def test_semantic_reranking(self):
        """Test semantic reranking functionality."""
        reranker = DocumentReranker(method="semantic")
        result = reranker.rerank_documents(self.query, self.documents)
        
        self.assertEqual(len(result), len(self.documents))
        
        # Check that semantic scores are added
        for doc in result:
            self.assertIn("semantic_score", doc)
            self.assertIsInstance(doc["semantic_score"], (int, float))
        
        # Results should be sorted by distance (lower distance = higher similarity)
        distances = [doc.get("distance", float('inf')) for doc in result]
        self.assertEqual(distances, sorted(distances))
    
    def test_hybrid_reranking(self):
        """Test hybrid reranking functionality."""
        reranker = DocumentReranker(method="hybrid")
        result = reranker.rerank_documents(self.query, self.documents)
        
        self.assertEqual(len(result), len(self.documents))
        
        # Check that all scores are added
        for doc in result:
            self.assertIn("hybrid_score", doc)
            self.assertIn("bm25_score", doc)
            self.assertIn("semantic_score", doc)
    
    def test_max_results_limit(self):
        """Test that max_results parameter works correctly."""
        reranker = DocumentReranker(method="bm25")
        result = reranker.rerank_documents(self.query, self.documents, max_results=2)
        
        self.assertEqual(len(result), 2)
    
    def test_empty_documents(self):
        """Test handling of empty document list."""
        reranker = DocumentReranker(method="bm25")
        result = reranker.rerank_documents(self.query, [])
        
        self.assertEqual(result, [])
    
    def test_custom_config(self):
        """Test reranker with custom configuration."""
        config = {
            "bm25_k1": 2.0,
            "bm25_b": 0.5,
            "vector_weight": 0.8,
            "bm25_weight": 0.2
        }
        reranker = DocumentReranker(method="hybrid", config=config)
        
        self.assertEqual(reranker.k1, 2.0)
        self.assertEqual(reranker.b, 0.5)
        self.assertEqual(reranker.vector_weight, 0.8)
        self.assertEqual(reranker.bm25_weight, 0.2)
    
    def test_factory_function(self):
        """Test the create_reranker factory function."""
        reranker = create_reranker(method="bm25")
        self.assertIsInstance(reranker, DocumentReranker)
        self.assertEqual(reranker.method, "bm25")
    
    def test_tokenization(self):
        """Test the tokenization method."""
        reranker = DocumentReranker()
        tokens = reranker._tokenize("Hello, world! This is a test.")
        
        expected_tokens = ["hello", "world", "this", "is", "test"]
        self.assertEqual(tokens, expected_tokens)
    
    def test_error_handling(self):
        """Test error handling in reranking."""
        # Test with malformed documents
        malformed_docs = [{"invalid": "structure"}]
        reranker = DocumentReranker(method="bm25")
        
        # Should not raise an exception and return original docs
        result = reranker.rerank_documents(self.query, malformed_docs)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
