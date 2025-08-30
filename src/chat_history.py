"""
Chat history management for conversational knowledge base interactions.

This module provides functionality to maintain conversation context across
multiple queries, enabling more natural and contextual interactions.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ChatHistory:
    """Manages conversation history for contextual interactions."""
    
    def __init__(self, session_id: Optional[str] = None, max_history: int = 10,
                 history_dir: str = ".chat_history"):
        """
        Initialize chat history manager.
        
        Args:
            session_id: Unique identifier for this conversation session
            max_history: Maximum number of exchanges to keep in memory
            history_dir: Directory to store chat history files
        """
        self.session_id = session_id or self._generate_session_id()
        self.max_history = max_history
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        
        self.history: List[Dict[str, Any]] = []
        self._load_history()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = int(time.time())
        return f"chat_{timestamp}"
    
    def _get_history_file(self) -> Path:
        """Get the path to the history file for this session."""
        return self.history_dir / f"{self.session_id}.json"
    
    def _load_history(self):
        """Load conversation history from file."""
        history_file = self._get_history_file()
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.history = data.get('exchanges', [])
                logger.info(f"Loaded {len(self.history)} conversation exchanges")
            except Exception as e:
                logger.error(f"Failed to load chat history: {e}")
                self.history = []
    
    def _save_history(self):
        """Save conversation history to file."""
        history_file = self._get_history_file()
        try:
            data = {
                'session_id': self.session_id,
                'created': datetime.now().isoformat(),
                'exchanges': self.history[-self.max_history:]  # Keep only recent exchanges
            }
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def add_exchange(self, user_query: str, assistant_response: str, 
                    context_docs: Optional[List[Dict]] = None,
                    metadata: Optional[Dict] = None):
        """
        Add a new question-answer exchange to the history.
        
        Args:
            user_query: The user's question/query
            assistant_response: The assistant's response
            context_docs: Documents used to generate the response
            metadata: Additional metadata (scores, timing, etc.)
        """
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'assistant_response': assistant_response,
            'context_docs': context_docs or [],
            'metadata': metadata or {}
        }
        
        self.history.append(exchange)
        
        # Keep only the most recent exchanges in memory
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self._save_history()
    
    def get_conversation_context(self, include_responses: bool = True,
                               max_exchanges: Optional[int] = None) -> str:
        """
        Get conversation context as a formatted string.
        
        Args:
            include_responses: Whether to include assistant responses
            max_exchanges: Maximum number of recent exchanges to include
            
        Returns:
            Formatted conversation context
        """
        if not self.history:
            return ""
        
        recent_history = self.history[-(max_exchanges or len(self.history)):]
        context_parts = []
        
        for i, exchange in enumerate(recent_history, 1):
            context_parts.append(f"Previous Question {i}: {exchange['user_query']}")
            if include_responses:
                # Truncate long responses for context
                response = exchange['assistant_response']
                if len(response) > 200:
                    response = response[:200] + "..."
                context_parts.append(f"Previous Answer {i}: {response}")
        
        return "\\n".join(context_parts)
    
    def get_related_queries(self, current_query: str, similarity_threshold: float = 0.3) -> List[str]:
        """
        Find previous queries that might be related to the current one.
        
        Args:
            current_query: The current user query
            similarity_threshold: Minimum similarity to consider related
            
        Returns:
            List of related previous queries
        """
        related_queries = []
        current_words = set(current_query.lower().split())
        
        for exchange in self.history:
            previous_query = exchange['user_query']
            previous_words = set(previous_query.lower().split())
            
            # Simple word overlap similarity
            if previous_words and current_words:
                overlap = len(current_words.intersection(previous_words))
                similarity = overlap / len(current_words.union(previous_words))
                
                if similarity >= similarity_threshold:
                    related_queries.append(previous_query)
        
        return related_queries
    
    def enhance_query_with_context(self, query: str, max_context_exchanges: int = 3) -> str:
        """
        Enhance the current query with conversation context.
        
        Args:
            query: The current user query
            max_context_exchanges: Maximum number of previous exchanges to include
            
        Returns:
            Enhanced query with conversation context
        """
        if not self.history:
            return query
        
        # Get recent conversation context
        context = self.get_conversation_context(
            include_responses=False,
            max_exchanges=max_context_exchanges
        )
        
        if context:
            enhanced_query = f"""Given the conversation context:
{context}

Current question: {query}

Please answer the current question considering the conversation history."""
            return enhanced_query
        
        return query
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dictionary with conversation statistics and info
        """
        if not self.history:
            return {
                'session_id': self.session_id,
                'total_exchanges': 0,
                'topics': [],
                'duration': '0 minutes'
            }
        
        first_exchange = self.history[0]
        last_exchange = self.history[-1]
        
        # Calculate duration
        start_time = datetime.fromisoformat(first_exchange['timestamp'])
        end_time = datetime.fromisoformat(last_exchange['timestamp'])
        duration = end_time - start_time
        
        # Extract common topics (simple word frequency)
        all_queries = " ".join([ex['user_query'] for ex in self.history])
        words = all_queries.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'session_id': self.session_id,
            'total_exchanges': len(self.history),
            'topics': [word for word, count in topics],
            'duration': str(duration).split('.')[0],  # Remove microseconds
            'start_time': first_exchange['timestamp'],
            'last_time': last_exchange['timestamp']
        }
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
        self._save_history()
    
    def export_history(self, format: str = "json") -> str:
        """
        Export conversation history.
        
        Args:
            format: Export format ("json" or "markdown")
            
        Returns:
            Formatted conversation history
        """
        if format == "markdown":
            lines = [f"# Conversation History - {self.session_id}\\n"]
            
            for i, exchange in enumerate(self.history, 1):
                timestamp = exchange['timestamp']
                lines.append(f"## Exchange {i} - {timestamp}\\n")
                lines.append(f"**User:** {exchange['user_query']}\\n")
                lines.append(f"**Assistant:** {exchange['assistant_response']}\\n")
                
                if exchange.get('context_docs'):
                    lines.append(f"**Sources:** {len(exchange['context_docs'])} documents\\n")
                lines.append("")
            
            return "\\n".join(lines)
        
        else:  # JSON format
            return json.dumps({
                'session_id': self.session_id,
                'exported': datetime.now().isoformat(),
                'exchanges': self.history
            }, indent=2, ensure_ascii=False)


def create_chat_history(session_id: Optional[str] = None, **kwargs) -> ChatHistory:
    """
    Factory function to create a ChatHistory instance.
    
    Args:
        session_id: Optional session identifier
        **kwargs: Additional arguments for ChatHistory constructor
        
    Returns:
        ChatHistory instance
    """
    return ChatHistory(session_id=session_id, **kwargs)
