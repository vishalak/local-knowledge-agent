import unittest
import sys
import os
from pathlib import Path

# Add src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from document_processors import extract_text_from_file

class TestDocumentProcessors(unittest.TestCase):

    def setUp(self):
        """Set up test files."""
        self.test_dir = Path("test_docs")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a simple text file
        self.text_file = self.test_dir / "test.txt"
        self.text_file.write_text("This is a test text file.")
        
        # Create a markdown file
        self.md_file = self.test_dir / "test.md"
        self.md_file.write_text("# Test Markdown\n\nThis is markdown content.")
        
        # Create a Python file
        self.py_file = self.test_dir / "test.py"
        self.py_file.write_text("# This is a Python comment\nprint('Hello, World!')")

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_text_from_txt(self):
        """Test extracting text from a .txt file."""
        result = extract_text_from_file(self.text_file)
        self.assertIsNotNone(result)
        self.assertEqual(result, "This is a test text file.")

    def test_extract_text_from_md(self):
        """Test extracting text from a .md file."""
        result = extract_text_from_file(self.md_file)
        self.assertIsNotNone(result)
        self.assertIn("Test Markdown", result)
        self.assertIn("markdown content", result)

    def test_extract_text_from_py(self):
        """Test extracting text from a .py file."""
        result = extract_text_from_file(self.py_file)
        self.assertIsNotNone(result)
        self.assertIn("Python comment", result)
        self.assertIn("Hello, World!", result)

    def test_extract_text_from_nonexistent_file(self):
        """Test handling of non-existent files."""
        nonexistent = self.test_dir / "nonexistent.txt"
        result = extract_text_from_file(nonexistent)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
