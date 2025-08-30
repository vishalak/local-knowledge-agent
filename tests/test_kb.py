import unittest
import sys
import os
import shutil
from pathlib import Path
import yaml

# Add src directory to the Python path to import the KnowledgeBase class
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from kb import KnowledgeBase

class TestKnowledgeBase(unittest.TestCase):

    def setUp(self):
        """Set up a temporary environment for testing."""
        self.test_dir = Path("test_temp_data")
        self.test_dir.mkdir(exist_ok=True)
        
        self.source_dir = self.test_dir / "source"
        self.source_dir.mkdir(exist_ok=True)
        
        self.chroma_path = str(self.test_dir / "chroma_db")
        
        # Create a dummy config file
        self.config_path = self.test_dir / "config.yaml"
        self.config = {
            "source_dir": str(self.source_dir),
            "chroma_path": self.chroma_path,
            "collection": "test_kb",
            "include_extensions": [".txt"],
            "excludes": ["exclude_this"],
            "chunk": {"size": 100, "overlap": 20, "semantic": False},
            "metadata": {"extract_advanced": True, "generate_summaries": False},
            "model": {"embedder": "BAAI/bge-small-en-v1.5", "llm": "llama3:8b"}
        }
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
            
        # Create some dummy source files
        (self.source_dir / "file1.txt").write_text("This is the first test file. It contains simple text.")
        (self.source_dir / "file2.txt").write_text("This is the second file, used for testing retrieval.")
        (self.source_dir / "document.md").write_text("This is a markdown file and should be ignored.")
        (self.source_dir / "exclude_this").mkdir(exist_ok=True)
        (self.source_dir / "exclude_this" / "excluded.txt").write_text("This file should be excluded.")

    def tearDown(self):
        """Clean up the temporary environment."""
        # Reset the client to release the file lock on Windows
        try:
            kb = KnowledgeBase(config_path=str(self.config_path))
            kb._reset_client()
        except Exception:
            pass
            
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_01_initialization(self):
        """Test that the KnowledgeBase class initializes correctly."""
        kb = KnowledgeBase(config_path=str(self.config_path))
        self.assertIsNotNone(kb)
        self.assertEqual(kb.cfg["collection"], "test_kb")
        self.assertIsNotNone(kb.collection)

    def test_02_chunk_text(self):
        """Test the text chunking logic."""
        kb = KnowledgeBase(config_path=str(self.config_path))
        text = "a" * 250
        # Use a dummy path that won't trigger code-aware chunking
        dummy_path = Path("test.txt")
        chunks = kb._chunk_text(text, dummy_path)
        
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][2], "a" * 100) # First chunk: 0-100
        self.assertEqual(chunks[1][2], "a" * 100) # Second chunk: 80-180 (with overlap)
        self.assertEqual(len(chunks[2][2]), 90)   # Third chunk: 160-250 (90 chars)
        self.assertTrue(chunks[1][0] < chunks[0][1]) # Check overlap exists

    def test_03_iter_files(self):
        """Test the file iteration logic."""
        kb = KnowledgeBase(config_path=str(self.config_path))
        files = list(kb._iter_files())
        filenames = [f.name for f in files]
        self.assertEqual(len(files), 2)
        self.assertIn("file1.txt", filenames)
        self.assertIn("file2.txt", filenames)
        self.assertNotIn("document.md", filenames)
        self.assertNotIn("excluded.txt", filenames)

    def test_04_build_and_retrieve(self):
        """Test building the KB and retrieving context."""
        kb = KnowledgeBase(config_path=str(self.config_path))
        
        # Build the knowledge base
        kb.build()
        
        # Check that the database and state file were created
        self.assertTrue(Path(self.chroma_path).exists())
        self.assertTrue(kb.state_path.exists())
        
        # Check the state file content
        state = kb._load_state()
        self.assertEqual(len(state["files"]), 2)
        self.assertIn(str(self.source_dir / "file1.txt"), state["files"])
        
        # Test retrieval
        query = "second file"
        items = kb.retrieve_context(query, k=1)
        self.assertEqual(len(items), 1)
        self.assertIn("second file", items[0]["text"])
        self.assertEqual(Path(items[0]["path"]).name, "file2.txt")

if __name__ == '__main__':
    unittest.main()
