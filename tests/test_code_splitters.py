import unittest
import sys
import os
from pathlib import Path

# Add src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from code_splitters import split_python_code, split_javascript_code, get_code_aware_chunks

class TestCodeSplitters(unittest.TestCase):

    def test_split_python_code(self):
        """Test Python code splitting based on functions and classes."""
        python_code = '''def function1():
    print("This is function 1")
    return True

class MyClass:
    def method1(self):
        print("This is a method")
    
    def method2(self):
        print("This is another method")

def function2():
    print("This is function 2")
    return False'''
        
        chunks = split_python_code(python_code, max_chunk_size=200)
        self.assertGreater(len(chunks), 1)  # Should split into multiple chunks
        
        # Check that function definitions start new chunks
        first_chunk = chunks[0][2]
        self.assertTrue(first_chunk.startswith('def function1():'))

    def test_split_javascript_code(self):
        """Test JavaScript code splitting."""
        js_code = '''function myFunction() {
    console.log("Hello, world!");
    return true;
}

class MyClass {
    constructor() {
        this.value = 0;
    }
    
    method() {
        console.log("Method called");
    }
}

const arrow = () => {
    console.log("Arrow function");
};'''
        
        chunks = split_javascript_code(js_code, max_chunk_size=200)
        self.assertGreater(len(chunks), 1)
        
        # Check that function definitions are preserved
        first_chunk = chunks[0][2]
        self.assertTrue('function myFunction()' in first_chunk)

    def test_get_code_aware_chunks_python(self):
        """Test code-aware chunking for Python files."""
        python_file = Path("test.py")
        python_code = '''def hello():
    print("Hello!")

def world():
    print("World!")'''
    
        chunks = get_code_aware_chunks(python_file, python_code)
        self.assertIsNotNone(chunks)
        self.assertGreater(len(chunks), 0)

    def test_get_code_aware_chunks_text(self):
        """Test that non-code files return None (fall back to regular chunking)."""
        text_file = Path("test.txt")
        text_content = "This is just regular text content."
        
        chunks = get_code_aware_chunks(text_file, text_content)
        self.assertIsNone(chunks)  # Should fall back to regular chunking

if __name__ == '__main__':
    unittest.main()
