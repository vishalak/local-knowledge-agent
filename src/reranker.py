"""
Document reranking module for improving retrieval precision.

This module provides various reranking strategies to reorder retrieved documents
based on their relevance to the query, improving the quality of results.
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DocumentReranker:
    """Reranks retrieved documents to improve relevance."""
    
    def __init__(self, method: str = "bm25", config: Optional[Dict] = None):
        """
        Initialize the reranker.
        
        Args:
            method: Reranking method ("bm25", "semantic", "hybrid", "none")
            config: Configuration parameters
        """
        self.method = method
        self.config = config or {}
        
        # BM25 parameters
        self.k1 = self.config.get('bm25_k1', 1.2)
        self.b = self.config.get('bm25_b', 0.75)
        
        # Hybrid weights
        self.vector_weight = self.config.get('vector_weight', 0.7)
        self.bm25_weight = self.config.get('bm25_weight', 0.3)
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], 
                        max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on the selected method.
        
        Args:
            query: The search query
            documents: List of document dictionaries with content and metadata
            max_results: Maximum number of results to return
            
        Returns:
            Reranked list of documents
        """
        if not documents or self.method == "none":
            return documents[:max_results] if max_results else documents
        
        try:
            if self.method == "bm25":
                ranked_docs = self._bm25_rerank(query, documents)
            elif self.method == "semantic":
                ranked_docs = self._semantic_rerank(query, documents)
            elif self.method == "hybrid":
                ranked_docs = self._hybrid_rerank(query, documents)
            else:
                logger.warning(f"Unknown reranking method: {self.method}")
                ranked_docs = documents
            
            return ranked_docs[:max_results] if max_results else ranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:max_results] if max_results else documents
    
    def _bm25_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank using BM25 algorithm."""
        if not documents:
            return documents
        
        # Tokenize query and documents
        query_terms = self._tokenize(query.lower())
        doc_terms = [self._tokenize(doc.get('content', '').lower()) for doc in documents]
        
        # Calculate document frequencies
        doc_freqs = {}
        total_docs = len(documents)
        
        for terms in doc_terms:
            unique_terms = set(terms)
            for term in unique_terms:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        
        # Calculate average document length
        avg_doc_length = sum(len(terms) for terms in doc_terms) / len(doc_terms) if doc_terms else 0
        
        # Calculate BM25 scores
        scores = []
        for i, (doc, terms) in enumerate(zip(documents, doc_terms)):
            score = self._calculate_bm25_score(
                query_terms, terms, doc_freqs, total_docs, avg_doc_length
            )
            scores.append((score, i, doc))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Update documents with BM25 scores
        ranked_docs = []
        for score, _, doc in scores:
            doc_copy = doc.copy()
            doc_copy['bm25_score'] = score
            ranked_docs.append(doc_copy)
        
        return ranked_docs
    
    def _semantic_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank using semantic similarity (uses existing vector scores)."""
        # Sort by existing vector similarity scores (distance field)
        # ChromaDB returns distances (lower is better), so we sort ascending
        try:
            sorted_docs = sorted(
                documents,
                key=lambda x: x.get('distance', float('inf'))
            )
            
            # Add semantic scores (convert distance to similarity)
            for doc in sorted_docs:
                distance = doc.get('distance', 1.0)
                # Convert distance to similarity score (0-1, higher is better)
                doc['semantic_score'] = max(0, 1 - distance)
            
            return sorted_docs
            
        except Exception as e:
            logger.error(f"Semantic reranking failed: {e}")
            return documents
    
    def _hybrid_rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank using hybrid approach (BM25 + semantic)."""
        # Get BM25 scores
        bm25_docs = self._bm25_rerank(query, documents)
        bm25_scores = {i: doc.get('bm25_score', 0) for i, doc in enumerate(bm25_docs)}
        
        # Get semantic scores
        semantic_docs = self._semantic_rerank(query, documents)
        semantic_scores = {i: doc.get('semantic_score', 0) for i, doc in enumerate(semantic_docs)}
        
        # Normalize scores to 0-1 range
        bm25_values = list(bm25_scores.values())
        semantic_values = list(semantic_scores.values())
        
        bm25_max = max(bm25_values) if bm25_values else 1
        bm25_min = min(bm25_values) if bm25_values else 0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1
        
        semantic_max = max(semantic_values) if semantic_values else 1
        semantic_min = min(semantic_values) if semantic_values else 0
        semantic_range = semantic_max - semantic_min if semantic_max != semantic_min else 1
        
        # Calculate hybrid scores
        hybrid_scores = []
        for i, doc in enumerate(documents):
            bm25_norm = (bm25_scores.get(i, 0) - bm25_min) / bm25_range
            semantic_norm = (semantic_scores.get(i, 0) - semantic_min) / semantic_range
            
            hybrid_score = (
                self.vector_weight * semantic_norm +
                self.bm25_weight * bm25_norm
            )
            
            doc_copy = doc.copy()
            doc_copy['hybrid_score'] = hybrid_score
            doc_copy['bm25_score'] = bm25_scores.get(i, 0)
            doc_copy['semantic_score'] = semantic_scores.get(i, 0)
            
            hybrid_scores.append((hybrid_score, doc_copy))
        
        # Sort by hybrid score (descending)
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for _, doc in hybrid_scores]
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_terms: List[str],
                             doc_freqs: Dict[str, int], total_docs: int,
                             avg_doc_length: float) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_length = len(doc_terms)
        term_counts = Counter(doc_terms)
        
        for term in query_terms:
            if term in term_counts:
                # Term frequency in document
                tf = term_counts[term]
                
                # Document frequency
                df = doc_freqs.get(term, 0)
                if df == 0:
                    continue
                
                # Inverse document frequency
                idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                
                # BM25 score component for this term
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                )
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (split on whitespace and punctuation)."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if len(token) > 1]  # Filter short tokens


def create_reranker(method: str = "bm25", config: Optional[Dict] = None) -> DocumentReranker:
    """
    Factory function to create a document reranker.
    
    Args:
        method: Reranking method ("bm25", "semantic", "hybrid", "none")
        config: Configuration parameters
        
    Returns:
        DocumentReranker instance
    """
    return DocumentReranker(method=method, config=config)
