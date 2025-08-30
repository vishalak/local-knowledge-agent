#!/usr/bin/env python3
"""
Code-aware text splitters for different programming languages.
These splitters create more meaningful chunks based on code structure.
"""

import re
from pathlib import Path
from typing import List, Tuple

def split_python_code(code: str, max_chunk_size: int = 1200) -> List[Tuple[int, int, str]]:
    """Split Python code into meaningful chunks based on classes and functions."""
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    current_start = 0
    current_size = 0
    
    for i, line in enumerate(lines):
        # Check if this line starts a new function or class
        if re.match(r'^(class|def)\s+', line.strip()):
            # If we have content and this would make the chunk too big, finalize the current chunk
            if current_chunk and current_size + len(line) > max_chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append((current_start, current_start + len(chunk_text), chunk_text))
                current_chunk = []
                current_start = sum(len(l) + 1 for l in lines[:i])  # +1 for newline
                current_size = 0
        
        current_chunk.append(line)
        current_size += len(line) + 1  # +1 for newline
        
        # If chunk is getting too big, finalize it
        if current_size > max_chunk_size:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((current_start, current_start + len(chunk_text), chunk_text))
            current_chunk = []
            current_start = sum(len(l) + 1 for l in lines[:i+1])
            current_size = 0
    
    # Add the final chunk if it has content
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append((current_start, current_start + len(chunk_text), chunk_text))
    
    return chunks

def split_javascript_code(code: str, max_chunk_size: int = 1200) -> List[Tuple[int, int, str]]:
    """Split JavaScript/TypeScript code into meaningful chunks based on functions and classes."""
    lines = code.split('\n')
    chunks = []
    current_chunk = []
    current_start = 0
    current_size = 0
    
    for i, line in enumerate(lines):
        # Check for function declarations, class declarations, etc.
        if re.match(r'^\s*(function|class|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=)\s+', line.strip()):
            if current_chunk and current_size + len(line) > max_chunk_size:
                chunk_text = '\n'.join(current_chunk)
                chunks.append((current_start, current_start + len(chunk_text), chunk_text))
                current_chunk = []
                current_start = sum(len(l) + 1 for l in lines[:i])
                current_size = 0
        
        current_chunk.append(line)
        current_size += len(line) + 1
        
        if current_size > max_chunk_size:
            chunk_text = '\n'.join(current_chunk)
            chunks.append((current_start, current_start + len(chunk_text), chunk_text))
            current_chunk = []
            current_start = sum(len(l) + 1 for l in lines[:i+1])
            current_size = 0
    
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        chunks.append((current_start, current_start + len(chunk_text), chunk_text))
    
    return chunks

def get_code_aware_chunks(file_path: Path, content: str, max_chunk_size: int = 1200) -> List[Tuple[int, int, str]]:
    """
    Get code-aware chunks based on file extension.
    Falls back to regular chunking for non-code files.
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.py':
        return split_python_code(content, max_chunk_size)
    elif suffix in ['.js', '.jsx', '.ts', '.tsx']:
        return split_javascript_code(content, max_chunk_size)
    else:
        # Fall back to regular character-based chunking
        return None  # Let the caller handle regular chunking
