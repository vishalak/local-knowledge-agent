import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

class TestMetadataExtractor(unittest.TestCase):

    def setUp(self):
        """Set up test files."""
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create test files with different types
        self.py_file = self.test_dir / "test_script.py"
        self.py_file.write_text('''#!/usr/bin/env python3
"""A test Python script."""

import os
import sys

class TestClass:
    def __init__(self):
        self.value = 42
    
    def method1(self):
        return "hello"

def main():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    main()
''')
        
        self.js_file = self.test_dir / "test_script.js"
        self.js_file.write_text('''// A test JavaScript file
import React from 'react';

class MyComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }
    
    render() {
        return <div>Count: {this.state.count}</div>;
    }
}

function helper() {
    return "helper function";
}

const arrow = () => {
    console.log("arrow function");
};

export default MyComponent;
''')
        
        self.md_file = self.test_dir / "README.md"
        self.md_file.write_text('''# Test Documentation

This is a test markdown file.

## Section 1

Some content here.

### Subsection 1.1

- Item 1
- Item 2
- Item 3

## Section 2

More content with a code block:

```python
def example():
    return "code"
```

That's all!
''')

    def tearDown(self):
        """Clean up test files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_file_metadata(self):
        """Test basic file metadata extraction."""
        try:
            from metadata_extractor import extract_file_metadata
            
            metadata = extract_file_metadata(self.py_file)
            
            # Check basic file info
            self.assertEqual(metadata["filename"], "test_script.py")
            self.assertEqual(metadata["file_extension"], ".py")
            self.assertIn("file_size", metadata)
            self.assertIn("created_time", metadata)
            self.assertIn("modified_time", metadata)
            self.assertTrue(metadata["is_code_file"])
            self.assertFalse(metadata["is_documentation"])
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

    def test_extract_python_content_metadata(self):
        """Test Python-specific content metadata extraction."""
        try:
            from metadata_extractor import extract_content_metadata
            
            text = self.py_file.read_text()
            metadata = extract_content_metadata(text, self.py_file)
            
            # Check content stats
            self.assertIn("line_count", metadata)
            self.assertIn("character_count", metadata)
            self.assertIn("word_count", metadata)
            
            # Check Python-specific metadata
            self.assertIn("python_functions", metadata)
            self.assertIn("python_classes", metadata)
            self.assertEqual(metadata["python_classes"], 1)  # TestClass
            self.assertGreaterEqual(metadata["python_functions"], 1)  # main function
            self.assertTrue(metadata.get("has_main_block", False))
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

    def test_extract_javascript_content_metadata(self):
        """Test JavaScript-specific content metadata extraction."""
        try:
            from metadata_extractor import extract_content_metadata
            
            text = self.js_file.read_text()
            metadata = extract_content_metadata(text, self.js_file)
            
            # Check JavaScript-specific metadata
            self.assertIn("js_functions", metadata)
            self.assertIn("js_classes", metadata)
            self.assertEqual(metadata["js_classes"], 1)  # MyComponent
            self.assertGreaterEqual(metadata["js_functions"], 2)  # helper, arrow
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

    def test_extract_markdown_content_metadata(self):
        """Test Markdown-specific content metadata extraction."""
        try:
            from metadata_extractor import extract_content_metadata
            
            text = self.md_file.read_text()
            metadata = extract_content_metadata(text, self.md_file)
            
            # Check Markdown-specific metadata
            self.assertIn("markdown_headers", metadata)
            self.assertIn("markdown_list_items", metadata)
            self.assertIn("markdown_code_blocks", metadata)
            self.assertGreaterEqual(metadata["markdown_headers"], 3)  # We have # ## ###
            self.assertGreaterEqual(metadata["markdown_list_items"], 3)  # Three list items
            self.assertGreaterEqual(metadata["markdown_code_blocks"], 1)  # One code block
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

    def test_extract_all_metadata(self):
        """Test comprehensive metadata extraction."""
        try:
            from metadata_extractor import extract_all_metadata
            
            text = self.py_file.read_text()
            metadata = extract_all_metadata(self.py_file, text, generate_summaries=False)
            
            # Should include both file and content metadata
            self.assertIn("filename", metadata)
            self.assertIn("is_code_file", metadata)
            self.assertIn("line_count", metadata)
            self.assertIn("python_functions", metadata)
            
            # Summary should not be included when generate_summaries=False
            self.assertNotIn("summary", metadata)
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

    def test_file_type_classification(self):
        """Test file type classification."""
        try:
            from metadata_extractor import extract_file_metadata
            
            py_meta = extract_file_metadata(self.py_file)
            js_meta = extract_file_metadata(self.js_file) 
            md_meta = extract_file_metadata(self.md_file)
            
            self.assertTrue(py_meta["is_code_file"])
            self.assertFalse(py_meta["is_documentation"])
            
            self.assertTrue(js_meta["is_code_file"])
            self.assertFalse(js_meta["is_documentation"])
            
            self.assertFalse(md_meta["is_code_file"])
            self.assertTrue(md_meta["is_documentation"])
            
        except ImportError:
            self.skipTest("Metadata extractor dependencies not available")

if __name__ == '__main__':
    unittest.main()
