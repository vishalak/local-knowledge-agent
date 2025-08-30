import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from code_splitters import split_csharp_code
from pathlib import Path


class TestCSharpCodeSplitter(unittest.TestCase):
    
    def setUp(self):
        """Set up test C# code."""
        self.sample_csharp = """using System;
using System.Collections.Generic;

namespace TestNamespace
{
    public class TestClass
    {
        private string _name;
        
        public string Name 
        { 
            get { return _name; } 
            set { _name = value; } 
        }
        
        public TestClass(string name)
        {
            _name = name;
        }
        
        public void DoSomething()
        {
            Console.WriteLine("Hello World");
        }
        
        public static void StaticMethod()
        {
            Console.WriteLine("Static method");
        }
    }
    
    public interface ITestInterface
    {
        void DoSomething();
    }
}"""
    
    def test_csharp_splitting(self):
        """Test that C# code is split appropriately."""
        chunks = split_csharp_code(self.sample_csharp, max_chunk_size=300)
        
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Each chunk should have start, end, and content
        for start, end, content in chunks:
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertIsInstance(content, str)
            self.assertGreater(end, start)
            self.assertGreater(len(content), 0)
    
    def test_csharp_preserves_structure(self):
        """Test that important C# constructs are preserved."""
        chunks = split_csharp_code(self.sample_csharp, max_chunk_size=500)
        
        all_text = "\n".join([chunk[2] for chunk in chunks])
        
        # Should preserve key C# keywords
        self.assertIn("namespace TestNamespace", all_text)
        self.assertIn("public class TestClass", all_text)
        self.assertIn("public interface ITestInterface", all_text)
        self.assertIn("public void DoSomething()", all_text)
    
    def test_empty_code(self):
        """Test handling of empty C# code."""
        chunks = split_csharp_code("", max_chunk_size=1200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][2], "")


if __name__ == "__main__":
    unittest.main()
