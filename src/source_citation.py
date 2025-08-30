#!/usr/bin/env python3
"""
Enhanced source citation system for the Knowledge Base.
Provides detailed, verifiable citations with context and metadata.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote


class SourceCitationManager:
    """Manages enhanced source citations with detailed context and metadata."""
    
    def __init__(self, source_dir: str):
        """
        Initialize the citation manager.
        
        Args:
            source_dir: Root directory of the source files
        """
        self.source_dir = Path(source_dir).expanduser()
    
    def format_sources_for_response(self, context_docs: List[Dict], 
                                  response_text: str = None) -> str:
        """
        Format sources for inclusion in the response with enhanced citations.
        
        Args:
            context_docs: List of context documents with metadata
            response_text: The generated response text (optional)
            
        Returns:
            Formatted citation text
        """
        if not context_docs:
            return ""
        
        citation_text = "\n\n---\n**Sources:**\n"
        
        for i, doc in enumerate(context_docs[:5], 1):  # Show top 5 sources
            citation = self._format_single_citation(doc, i)
            citation_text += f"\n{citation}"
        
        # Add summary
        total_sources = len(context_docs)
        if total_sources > 5:
            citation_text += f"\n\n*... and {total_sources - 5} more sources*"
        
        return citation_text
    
    def _format_single_citation(self, doc: Dict, index: int) -> str:
        """
        Format a single source citation with detailed information.
        
        Args:
            doc: Document metadata
            index: Citation index number
            
        Returns:
            Formatted citation string
        """
        # Extract basic information
        path = doc.get('path', 'Unknown')
        start_line = doc.get('start', 0)
        end_line = doc.get('end', 0)
        distance = doc.get('distance')
        
        # Create relative path for display
        try:
            rel_path = Path(path).relative_to(self.source_dir)
        except (ValueError, TypeError):
            rel_path = Path(path).name if path else "Unknown"
        
        # Build the citation
        citation = f"**[{index}]** `{rel_path}`"
        
        # Add line numbers if available
        if start_line and end_line:
            if start_line == end_line:
                citation += f" (line {start_line})"
            else:
                citation += f" (lines {start_line}-{end_line})"
        
        # Add relevance score
        if distance is not None:
            relevance = max(0, 1 - distance)  # Convert distance to relevance
            citation += f" - *Relevance: {relevance:.1%}*"
        
        # Add file metadata if available
        metadata_info = self._extract_metadata_info(doc)
        if metadata_info:
            citation += f"\n   {metadata_info}"
        
        # Add content preview
        preview = self._create_content_preview(doc)
        if preview:
            citation += f"\n   📄 *{preview}*"
        
        return citation
    
    def _extract_metadata_info(self, doc: Dict) -> str:
        """Extract and format relevant metadata information."""
        info_parts = []
        
        # File type
        if doc.get('is_code_file'):
            info_parts.append("📁 Code")
        elif doc.get('is_documentation'):
            info_parts.append("📚 Documentation")
        elif doc.get('is_config_file'):
            info_parts.append("⚙️ Configuration")
        
        # File size
        if doc.get('file_size_kb'):
            size_kb = doc.get('file_size_kb')
            if size_kb < 1:
                info_parts.append(f"📏 {doc.get('file_size', 0)} bytes")
            else:
                info_parts.append(f"📏 {size_kb} KB")
        
        # Last modified
        if doc.get('modified_date'):
            try:
                # Extract just the date part
                date_str = doc.get('modified_date', '').split('T')[0]
                info_parts.append(f"🕒 Modified: {date_str}")
            except:
                pass
        
        return " | ".join(info_parts)
    
    def _create_content_preview(self, doc: Dict, max_length: int = 100) -> str:
        """Create a preview of the document content."""
        text = doc.get('text', '')
        if not text:
            return ""
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= max_length:
            return text
        
        # Truncate and add ellipsis
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return f"{truncated}..."
    
    def generate_vscode_links(self, context_docs: List[Dict]) -> List[str]:
        """
        Generate VS Code links for opening files at specific lines.
        
        Args:
            context_docs: List of context documents
            
        Returns:
            List of VS Code URIs
        """
        links = []
        
        for doc in context_docs:
            path = doc.get('path')
            start_line = doc.get('start', 1)
            
            if path:
                try:
                    abs_path = Path(path).resolve()
                    # VS Code URI format: vscode://file/path:line:column
                    uri = f"vscode://file/{abs_path}:{start_line}:1"
                    links.append(uri)
                except Exception:
                    continue
        
        return links
    
    def format_sources_for_terminal(self, context_docs: List[Dict], 
                                  show_preview: bool = True,
                                  max_sources: int = 5) -> str:
        """
        Format sources for terminal display with rich formatting.
        
        Args:
            context_docs: List of context documents
            show_preview: Whether to show content preview
            max_sources: Maximum number of sources to display
            
        Returns:
            Rich-formatted string for terminal display
        """
        if not context_docs:
            return "[dim]No sources available[/dim]"
        
        sources_text = "[bold]Sources:[/bold]\n"
        
        for i, doc in enumerate(context_docs[:max_sources], 1):
            # Basic citation
            path = doc.get('path', 'Unknown')
            try:
                rel_path = Path(path).relative_to(self.source_dir)
            except (ValueError, TypeError):
                rel_path = Path(path).name if path else "Unknown"
            
            sources_text += f"\n[cyan][{i}][/cyan] [bold]{rel_path}[/bold]"
            
            # Line numbers
            start_line = doc.get('start')
            end_line = doc.get('end')
            if start_line and end_line:
                if start_line == end_line:
                    sources_text += f"[dim]:{start_line}[/dim]"
                else:
                    sources_text += f"[dim]:{start_line}-{end_line}[/dim]"
            
            # Relevance score
            distance = doc.get('distance')
            if distance is not None:
                relevance = max(0, 1 - distance)
                color = "green" if relevance > 0.8 else "yellow" if relevance > 0.6 else "red"
                sources_text += f" [dim]([{color}]{relevance:.1%}[/{color}])[/dim]"
            
            # Content preview
            if show_preview:
                preview = self._create_content_preview(doc, max_length=80)
                if preview:
                    sources_text += f"\n    [dim italic]{preview}[/dim italic]"
        
        # Show total count if truncated
        total_sources = len(context_docs)
        if total_sources > max_sources:
            sources_text += f"\n[dim]... and {total_sources - max_sources} more sources[/dim]"
        
        return sources_text
    
    def create_citation_map(self, context_docs: List[Dict], 
                          response_text: str) -> Dict[str, List[int]]:
        """
        Create a mapping of response sentences to source citations.
        
        Args:
            context_docs: List of context documents
            response_text: The generated response text
            
        Returns:
            Dictionary mapping sentence indices to source indices
        """
        # This is a placeholder for more sophisticated citation mapping
        # Could use semantic similarity between response sentences and source content
        citation_map = {}
        
        # Simple implementation: map based on keyword overlap
        sentences = self._split_into_sentences(response_text)
        
        for sent_idx, sentence in enumerate(sentences):
            matching_sources = []
            
            for src_idx, doc in enumerate(context_docs):
                # Simple keyword matching (could be enhanced with semantic similarity)
                doc_text = doc.get('text', '').lower()
                sentence_lower = sentence.lower()
                
                # Count word overlaps
                doc_words = set(re.findall(r'\w+', doc_text))
                sent_words = set(re.findall(r'\w+', sentence_lower))
                
                overlap = len(doc_words.intersection(sent_words))
                if overlap > 2:  # Threshold for considering a match
                    matching_sources.append(src_idx + 1)  # 1-indexed
            
            if matching_sources:
                citation_map[str(sent_idx)] = matching_sources
        
        return citation_map
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (could be enhanced with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def format_inline_citations(self, response_text: str, 
                              context_docs: List[Dict]) -> str:
        """
        Add inline citations to the response text.
        
        Args:
            response_text: The generated response
            context_docs: List of context documents
            
        Returns:
            Response text with inline citations
        """
        # Create citation map
        citation_map = self.create_citation_map(context_docs, response_text)
        
        # Split response into sentences
        sentences = self._split_into_sentences(response_text)
        
        # Add inline citations
        enhanced_sentences = []
        for i, sentence in enumerate(sentences):
            if str(i) in citation_map:
                citations = citation_map[str(i)]
                citation_str = "[" + ",".join(map(str, citations)) + "]"
                sentence = f"{sentence}{citation_str}"
            enhanced_sentences.append(sentence)
        
        return ". ".join(enhanced_sentences) + "."
