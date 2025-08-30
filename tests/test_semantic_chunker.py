import unittest
import sys
import os

# Add src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestSemanticChunker(unittest.TestCase):

    def test_sentence_splitting(self):
        """Test basic sentence splitting functionality."""
        try:
            from semantic_chunker import SemanticChunker
            
            chunker = SemanticChunker()
            text = "This is the first sentence. This is the second sentence! Is this the third sentence?"
            sentences = chunker._split_into_sentences(text)
            
            self.assertEqual(len(sentences), 3)
            self.assertEqual(sentences[0], "This is the first sentence.")
            self.assertEqual(sentences[1], "This is the second sentence!")
            self.assertEqual(sentences[2], "Is this the third sentence?")
        except ImportError:
            self.skipTest("Semantic chunking dependencies not available")

    def test_semantic_chunking_fallback(self):
        """Test that semantic chunking gracefully falls back when dependencies are missing."""
        try:
            from semantic_chunker import get_semantic_chunks
            
            text = "Machine learning is fascinating. Artificial intelligence transforms industries. Deep learning uses neural networks. Python is a programming language."
            chunks = get_semantic_chunks(text, max_chunk_size=100)
            
            # Should return either valid chunks or None (for fallback)
            self.assertTrue(chunks is None or isinstance(chunks, list))
            
            if chunks is not None:
                self.assertGreater(len(chunks), 0)
                # Each chunk should be a tuple with 3 elements
                for chunk in chunks:
                    self.assertEqual(len(chunk), 3)
                    self.assertIsInstance(chunk[0], int)  # start position
                    self.assertIsInstance(chunk[1], int)  # end position
                    self.assertIsInstance(chunk[2], str)  # chunk text
                    
        except ImportError:
            self.skipTest("Semantic chunking dependencies not available")

    def test_empty_text_handling(self):
        """Test that empty text is handled properly."""
        try:
            from semantic_chunker import get_semantic_chunks
            
            chunks = get_semantic_chunks("", max_chunk_size=100)
            # Should either return empty list or None
            self.assertTrue(chunks is None or chunks == [])
            
        except ImportError:
            self.skipTest("Semantic chunking dependencies not available")

if __name__ == '__main__':
    unittest.main()
