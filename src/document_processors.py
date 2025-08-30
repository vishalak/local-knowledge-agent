#!/usr/bin/env python3
"""
Document processors for different file types.
Each processor extracts text content from a specific file format.
"""

import os
from pathlib import Path
from typing import Optional

def extract_text_from_pdf(file_path: Path) -> Optional[str]:
    """Extract text from PDF files using PyPDF2."""
    try:
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except ImportError:
        print(f"[yellow]Warning: PyPDF2 not installed. Skipping PDF file: {file_path}[/yellow]")
        return None
    except Exception as e:
        print(f"[yellow]Warning: Failed to process PDF {file_path}: {e}[/yellow]")
        return None

def extract_text_from_docx(file_path: Path) -> Optional[str]:
    """Extract text from Word documents using python-docx."""
    try:
        from docx import Document
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except ImportError:
        print(f"[yellow]Warning: python-docx not installed. Skipping Word file: {file_path}[/yellow]")
        return None
    except Exception as e:
        print(f"[yellow]Warning: Failed to process Word document {file_path}: {e}[/yellow]")
        return None

def extract_text_from_file(file_path: Path) -> Optional[str]:
    """
    Extract text from various file types.
    Returns the extracted text or None if extraction fails.
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    elif suffix in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        # Handle text-based files
        try:
            text = file_path.read_text(encoding='utf-8')
            return text
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding='latin-1')
                return text
            except Exception as e:
                print(f"[yellow]Warning: Failed to read text file {file_path}: {e}[/yellow]")
                return None
        except Exception as e:
            print(f"[yellow]Warning: Failed to process file {file_path}: {e}[/yellow]")
            return None
