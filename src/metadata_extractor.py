#!/usr/bin/env python3
"""
Advanced metadata extraction for documents.
Extracts and processes metadata like timestamps, file info, and content summaries.
"""

import os
import stat
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

try:
    import ollama
except ImportError:
    ollama = None

def extract_file_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing file metadata
    """
    try:
        file_stats = file_path.stat()
        
        metadata = {
            # Basic file information
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_stats.st_size,
            "file_size_kb": round(file_stats.st_size / 1024, 2),
            
            # Timestamps
            "created_time": int(file_stats.st_ctime),
            "modified_time": int(file_stats.st_mtime),
            "accessed_time": int(file_stats.st_atime),
            
            # Human-readable timestamps
            "created_date": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_date": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "accessed_date": datetime.fromtimestamp(file_stats.st_atime).isoformat(),
            
            # File type classification
            "is_code_file": _is_code_file(file_path),
            "is_documentation": _is_documentation_file(file_path),
            "is_config_file": _is_config_file(file_path),
        }
        
        # Add file permissions if available (Unix-like systems)
        if hasattr(file_stats, 'st_mode'):
            metadata["file_permissions"] = stat.filemode(file_stats.st_mode)
        
        return metadata
        
    except Exception as e:
        print(f"[yellow]Warning: Failed to extract metadata for {file_path}: {e}[/yellow]")
        return {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "error": str(e)
        }

def _is_code_file(file_path: Path) -> bool:
    """Check if the file is a source code file."""
    code_extensions = {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.php', '.rb', '.go', '.rs', '.kt', '.swift', '.scala', '.r',
        '.m', '.mm', '.pl', '.sh', '.bash', '.ps1', '.sql', '.html', '.css'
    }
    return file_path.suffix.lower() in code_extensions

def _is_documentation_file(file_path: Path) -> bool:
    """Check if the file is a documentation file."""
    doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx', '.pdf', '.tex'}
    doc_patterns = ['readme', 'changelog', 'license', 'contributing', 'docs']
    
    if file_path.suffix.lower() in doc_extensions:
        return True
    
    filename_lower = file_path.name.lower()
    return any(pattern in filename_lower for pattern in doc_patterns)

def _is_config_file(file_path: Path) -> bool:
    """Check if the file is a configuration file."""
    config_extensions = {'.yaml', '.yml', '.json', '.ini', '.toml', '.cfg', '.conf', '.config'}
    config_patterns = ['config', 'settings', '.env', 'dockerfile', 'makefile']
    
    if file_path.suffix.lower() in config_extensions:
        return True
    
    filename_lower = file_path.name.lower()
    return any(pattern in filename_lower for pattern in config_patterns)

def extract_content_metadata(text: str, file_path: Path) -> Dict[str, Any]:
    """
    Extract metadata from the content of the file.
    
    Args:
        text: File content as text
        file_path: Path to the file
        
    Returns:
        Dictionary containing content-based metadata
    """
    if not text:
        return {}
    
    lines = text.split('\n')
    
    metadata = {
        "line_count": len(lines),
        "character_count": len(text),
        "word_count": len(text.split()),
        "non_empty_lines": len([line for line in lines if line.strip()]),
    }
    
    # Extract language-specific metadata
    if file_path.suffix.lower() == '.py':
        metadata.update(_extract_python_metadata(text))
    elif file_path.suffix.lower() in ['.js', '.jsx', '.ts', '.tsx']:
        metadata.update(_extract_javascript_metadata(text))
    elif file_path.suffix.lower() in ['.md', '.rst']:
        metadata.update(_extract_markdown_metadata(text))
    
    return metadata

def _extract_python_metadata(text: str) -> Dict[str, Any]:
    """Extract Python-specific metadata."""
    import re
    
    metadata = {}
    
    # Count functions and classes
    functions = re.findall(r'^def\s+(\w+)', text, re.MULTILINE)
    classes = re.findall(r'^class\s+(\w+)', text, re.MULTILINE)
    imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(.+)', text, re.MULTILINE)
    
    metadata.update({
        "python_functions": len(functions),
        "python_classes": len(classes),
        "python_imports": len(imports),
        "function_names": functions[:10],  # Store first 10 function names
        "class_names": classes[:10],       # Store first 10 class names
    })
    
    # Check for common patterns
    if 'if __name__ == "__main__"' in text:
        metadata["has_main_block"] = True
    if any(word in text for word in ['TODO', 'FIXME', 'XXX', 'HACK']):
        metadata["has_todo_comments"] = True
    
    return metadata

def _extract_javascript_metadata(text: str) -> Dict[str, Any]:
    """Extract JavaScript/TypeScript-specific metadata."""
    import re
    
    metadata = {}
    
    # Count functions and classes
    functions = re.findall(r'(?:function\s+(\w+)|(\w+)\s*(?:=|:)\s*(?:function|\([^)]*\)\s*=>))', text)
    classes = re.findall(r'class\s+(\w+)', text)
    imports = re.findall(r'import\s+.*?from\s+[\'"](.+?)[\'"]', text)
    
    # Flatten function matches (regex returns tuples)
    function_names = [f[0] or f[1] for f in functions if f[0] or f[1]]
    
    metadata.update({
        "js_functions": len(function_names),
        "js_classes": len(classes),
        "js_imports": len(imports),
        "function_names": function_names[:10],
        "class_names": classes[:10],
    })
    
    return metadata

def _extract_markdown_metadata(text: str) -> Dict[str, Any]:
    """Extract Markdown-specific metadata."""
    import re
    
    metadata = {}
    
    # Count headers
    headers = re.findall(r'^(#{1,6})\s+(.+)', text, re.MULTILINE)
    
    metadata.update({
        "markdown_headers": len(headers),
        "header_levels": [len(h[0]) for h in headers],
        "top_level_headers": [h[1] for h in headers if len(h[0]) <= 2][:5],
    })
    
    # Count lists and code blocks
    list_items = re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE)
    code_blocks = re.findall(r'```[\s\S]*?```', text)
    
    metadata.update({
        "markdown_list_items": len(list_items),
        "markdown_code_blocks": len(code_blocks),
    })
    
    return metadata

def generate_summary(text: str, max_length: int = 200, model: str = "llama3:8b") -> Optional[str]:
    """
    Generate a summary of the text content using Ollama.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary in characters
        model: Ollama model to use for summarization
        
    Returns:
        Generated summary or None if summarization fails
    """
    if not ollama or not text.strip():
        return None
    
    try:
        # Truncate text if it's too long for the model
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        prompt = f"""Please provide a concise summary of the following text in no more than {max_length} characters:

{text}

Summary:"""
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}  # Lower temperature for more consistent summaries
        )
        
        summary = response["message"]["content"].strip()
        
        # Truncate if still too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
        
    except Exception as e:
        print(f"[yellow]Warning: Failed to generate summary: {e}[/yellow]")
        return None

def extract_all_metadata(file_path: Path, text: str, generate_summaries: bool = False, 
                        llm_model: str = "llama3:8b") -> Dict[str, Any]:
    """
    Extract all available metadata for a file and its content.
    
    Args:
        file_path: Path to the file
        text: File content as text
        generate_summaries: Whether to generate AI summaries
        llm_model: Model to use for summary generation
        
    Returns:
        Complete metadata dictionary
    """
    metadata = {}
    
    # Extract file-based metadata
    metadata.update(extract_file_metadata(file_path))
    
    # Extract content-based metadata
    if text:
        metadata.update(extract_content_metadata(text, file_path))
        
        # Generate summary if requested
        if generate_summaries:
            summary = generate_summary(text, model=llm_model)
            if summary:
                metadata["summary"] = summary
    
    return metadata
