#!/usr/bin/env python3
"""
Semantic text chunking based on sentence similarity.
Groups related sentences together rather than using fixed character counts.
"""

import re
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", similarity_threshold: float = 0.7):
        """
        Initialize the semantic chunker.
        
        Args:
            model_name: Name of the sentence transformer model to use
            similarity_threshold: Similarity threshold for grouping sentences (0.0 to 1.0)
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter out empty sentences and strip whitespace
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_similar_sentences(self, sentences: List[str]) -> List[List[int]]:
        """Group sentences based on semantic similarity."""
        if len(sentences) <= 1:
            return [[0]] if sentences else []
        
        # Get embeddings for all sentences
        embeddings = self.model.encode(sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group sentences based on similarity
        groups = []
        visited = set()
        
        for i in range(len(sentences)):
            if i in visited:
                continue
                
            # Start a new group with the current sentence
            current_group = [i]
            visited.add(i)
            
            # Find similar sentences to add to this group
            for j in range(i + 1, len(sentences)):
                if j in visited:
                    continue
                    
                # Check if sentence j is similar to any sentence in the current group
                max_similarity = max(similarity_matrix[i][j] for i in current_group)
                
                if max_similarity >= self.similarity_threshold:
                    current_group.append(j)
                    visited.add(j)
            
            groups.append(current_group)
        
        return groups
    
    def chunk_text(self, text: str, max_chunk_size: int = 1200) -> List[Tuple[int, int, str]]:
        """
        Chunk text semantically while respecting size limits.
        
        Args:
            text: The text to chunk
            max_chunk_size: Maximum size for each chunk in characters
            
        Returns:
            List of tuples (start_pos, end_pos, chunk_text)
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []
        
        # Group sentences by semantic similarity
        sentence_groups = self._group_similar_sentences(sentences)
        
        chunks = []
        
        for group in sentence_groups:
            # Get the sentences in this semantic group
            group_sentences = [sentences[i] for i in sorted(group)]
            group_text = ' '.join(group_sentences)
            
            # If the group is small enough, make it a single chunk
            if len(group_text) <= max_chunk_size:
                # Find the start position of the first sentence in the group
                first_sentence = sentences[min(group)]
                start_pos = text.find(first_sentence)
                if start_pos == -1:
                    start_pos = 0
                
                end_pos = start_pos + len(group_text)
                chunks.append((start_pos, end_pos, group_text))
            else:
                # If the group is too large, split it further by size while keeping semantic grouping
                current_chunk = ""
                chunk_start = None
                
                for sentence_idx in sorted(group):
                    sentence = sentences[sentence_idx]
                    
                    # Check if adding this sentence would exceed the limit
                    if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                        # Finalize the current chunk
                        if chunk_start is None:
                            chunk_start = text.find(current_chunk.split('.')[0])
                        if chunk_start == -1:
                            chunk_start = 0
                        
                        chunk_end = chunk_start + len(current_chunk)
                        chunks.append((chunk_start, chunk_end, current_chunk.strip()))
                        
                        # Start a new chunk
                        current_chunk = sentence
                        chunk_start = text.find(sentence)
                    else:
                        # Add sentence to current chunk
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
                            chunk_start = text.find(sentence)
                
                # Add the final chunk if it has content
                if current_chunk:
                    if chunk_start is None:
                        chunk_start = 0
                    chunk_end = chunk_start + len(current_chunk)
                    chunks.append((chunk_start, chunk_end, current_chunk.strip()))
        
        return chunks

def get_semantic_chunks(text: str, max_chunk_size: int = 1200, similarity_threshold: float = 0.7) -> List[Tuple[int, int, str]]:
    """
    Convenience function to get semantic chunks.
    
    Args:
        text: The text to chunk
        max_chunk_size: Maximum chunk size in characters
        similarity_threshold: Similarity threshold for grouping sentences
        
    Returns:
        List of tuples (start_pos, end_pos, chunk_text)
    """
    try:
        chunker = SemanticChunker(similarity_threshold=similarity_threshold)
        return chunker.chunk_text(text, max_chunk_size)
    except Exception as e:
        print(f"[yellow]Warning: Semantic chunking failed ({e}), falling back to regular chunking[/yellow]")
        return None  # Signal to fall back to regular chunking
