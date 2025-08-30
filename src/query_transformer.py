#!/usr/bin/env python3
"""
Query transformation techniques to improve retrieval quality.
Includes HyDE (Hypothetical Document Embeddings) and query expansion.
"""

from typing import List, Optional, Dict, Any
import re

try:
    import ollama
except ImportError:
    ollama = None

class QueryTransformer:
    def __init__(self, llm_model: str = "llama3:8b"):
        """
        Initialize the query transformer.
        
        Args:
            llm_model: Name of the Ollama model to use for query transformation
        """
        self.llm_model = llm_model

    def generate_hyde_query(self, original_query: str) -> Optional[str]:
        """
        Generate a hypothetical document that would answer the query (HyDE technique).
        
        Args:
            original_query: The user's original question
            
        Returns:
            Hypothetical document text or None if generation fails
        """
        if not ollama:
            return None
            
        try:
            prompt = f"""Given the following question, write a hypothetical document passage that would contain the answer. Write as if you are an expert explaining the topic clearly and concisely.

Question: {original_query}

Hypothetical document passage:"""

            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            
            hyde_text = response["message"]["content"].strip()
            
            # Clean up the response - remove any meta-commentary
            hyde_text = re.sub(r'^(Here\'s|This is|The following is).*?:\s*', '', hyde_text, flags=re.IGNORECASE)
            
            return hyde_text
            
        except Exception as e:
            print(f"[yellow]Warning: HyDE generation failed: {e}[/yellow]")
            return None

    def expand_query(self, original_query: str) -> List[str]:
        """
        Expand the query with synonyms and related terms.
        
        Args:
            original_query: The user's original question
            
        Returns:
            List of expanded query variations
        """
        if not ollama:
            return [original_query]
            
        try:
            prompt = f"""Given the following query, provide 3-5 alternative ways to phrase the same question using different words, synonyms, and technical terms. Focus on different ways a developer or technical writer might express the same concept.

Original query: {original_query}

Alternative phrasings:
1."""

            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5}
            )
            
            expanded_text = response["message"]["content"].strip()
            
            # Extract numbered alternatives
            alternatives = []
            lines = expanded_text.split('\n')
            for line in lines:
                # Match numbered items (1., 2., etc.)
                match = re.match(r'^\d+\.\s*(.+)', line.strip())
                if match:
                    alternatives.append(match.group(1).strip())
            
            # Include the original query
            all_queries = [original_query] + alternatives
            return all_queries[:5]  # Limit to 5 total queries
            
        except Exception as e:
            print(f"[yellow]Warning: Query expansion failed: {e}[/yellow]")
            return [original_query]

    def enhance_query_with_context(self, query: str, context_hints: List[str] = None) -> str:
        """
        Enhance the query with additional context or domain-specific terms.
        
        Args:
            query: Original query
            context_hints: Optional list of context hints (e.g., file types, domains)
            
        Returns:
            Enhanced query string
        """
        enhanced_query = query
        
        # Add technical context if hints are provided
        if context_hints:
            context_str = " ".join(context_hints)
            enhanced_query = f"{query} (in context of: {context_str})"
        
        # Add common technical qualifiers for better matching
        tech_qualifiers = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['error', 'bug', 'problem', 'issue', 'fix']):
            tech_qualifiers.append("troubleshooting")
        
        if any(word in query_lower for word in ['how to', 'tutorial', 'guide', 'example']):
            tech_qualifiers.append("documentation")
            
        if any(word in query_lower for word in ['install', 'setup', 'configure', 'deploy']):
            tech_qualifiers.append("installation")
            
        if tech_qualifiers:
            enhanced_query += f" {' '.join(tech_qualifiers)}"
            
        return enhanced_query

    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from the query for focused searching.
        
        Args:
            query: The search query
            
        Returns:
            List of key terms
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'how', 'what', 'where', 'when', 'why',
            'can', 'could', 'should', 'would', 'do', 'does', 'did', 'i', 'me',
            'my', 'you', 'your', 'we', 'us', 'our'
        }
        
        # Extract words, keeping technical terms and identifiers
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Also extract quoted phrases and technical patterns
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        key_terms.extend(quoted_phrases)
        
        # Extract technical patterns (camelCase, snake_case, etc.)
        tech_patterns = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b\w+_\w+\b|\b[A-Z_]+\b', query)
        key_terms.extend(tech_patterns)
        
        return list(set(key_terms))  # Remove duplicates

def transform_query(original_query: str, method: str = "hyde", llm_model: str = "llama3:8b", 
                   context_hints: List[str] = None) -> Dict[str, Any]:
    """
    Apply query transformation using the specified method.
    
    Args:
        original_query: The user's original question
        method: Transformation method ("hyde", "expand", "enhance", "extract")
        llm_model: Ollama model to use
        context_hints: Optional context hints for enhancement
        
    Returns:
        Dictionary containing transformation results
    """
    transformer = QueryTransformer(llm_model)
    
    result = {
        "original_query": original_query,
        "method": method,
        "transformed_queries": [original_query]  # Always include original as fallback
    }
    
    if method == "hyde":
        hyde_text = transformer.generate_hyde_query(original_query)
        if hyde_text:
            result["hyde_document"] = hyde_text
            result["transformed_queries"] = [hyde_text, original_query]
        
    elif method == "expand":
        expanded_queries = transformer.expand_query(original_query)
        result["transformed_queries"] = expanded_queries
        
    elif method == "enhance":
        enhanced_query = transformer.enhance_query_with_context(original_query, context_hints)
        result["enhanced_query"] = enhanced_query
        result["transformed_queries"] = [enhanced_query]
        
    elif method == "extract":
        key_terms = transformer.extract_key_terms(original_query)
        result["key_terms"] = key_terms
        result["transformed_queries"] = [" ".join(key_terms), original_query]
        
    elif method == "multi":
        # Apply multiple techniques
        hyde_text = transformer.generate_hyde_query(original_query)
        expanded_queries = transformer.expand_query(original_query)
        enhanced_query = transformer.enhance_query_with_context(original_query, context_hints)
        
        all_queries = [original_query]
        if hyde_text:
            all_queries.append(hyde_text)
            result["hyde_document"] = hyde_text
        all_queries.extend(expanded_queries[1:])  # Skip original query from expansion
        if enhanced_query != original_query:
            all_queries.append(enhanced_query)
            
        result["transformed_queries"] = list(set(all_queries))[:6]  # Limit to 6 queries max
    
    return result
