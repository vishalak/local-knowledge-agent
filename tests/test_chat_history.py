import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat_history import ChatHistory, create_chat_history


class TestChatHistory(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.chat_history = ChatHistory(
            session_id="test_session",
            max_history=5,
            history_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test chat history initialization."""
        self.assertEqual(self.chat_history.session_id, "test_session")
        self.assertEqual(self.chat_history.max_history, 5)
        self.assertEqual(len(self.chat_history.history), 0)
    
    def test_add_exchange(self):
        """Test adding a conversation exchange."""
        query = "What is Python?"
        response = "Python is a programming language."
        
        self.chat_history.add_exchange(query, response)
        
        self.assertEqual(len(self.chat_history.history), 1)
        exchange = self.chat_history.history[0]
        self.assertEqual(exchange['user_query'], query)
        self.assertEqual(exchange['assistant_response'], response)
        self.assertIn('timestamp', exchange)
    
    def test_max_history_limit(self):
        """Test that history is limited to max_history."""
        # Add more exchanges than the limit
        for i in range(7):
            self.chat_history.add_exchange(f"Question {i}", f"Answer {i}")
        
        # Should only keep the most recent 5
        self.assertEqual(len(self.chat_history.history), 5)
        
        # Check that the most recent exchanges are kept
        self.assertEqual(self.chat_history.history[0]['user_query'], "Question 2")
        self.assertEqual(self.chat_history.history[-1]['user_query'], "Question 6")
    
    def test_get_conversation_context(self):
        """Test getting conversation context."""
        # Add some exchanges
        self.chat_history.add_exchange("First question", "First answer")
        self.chat_history.add_exchange("Second question", "Second answer")
        
        # Test context without responses
        context = self.chat_history.get_conversation_context(include_responses=False)
        self.assertIn("Previous Question 1: First question", context)
        self.assertIn("Previous Question 2: Second question", context)
        self.assertNotIn("First answer", context)
        
        # Test context with responses
        context_with_responses = self.chat_history.get_conversation_context(include_responses=True)
        self.assertIn("First answer", context_with_responses)
        self.assertIn("Second answer", context_with_responses)
    
    def test_enhance_query_with_context(self):
        """Test query enhancement with conversation context."""
        # Empty history should return original query
        query = "What is machine learning?"
        enhanced = self.chat_history.enhance_query_with_context(query)
        self.assertEqual(enhanced, query)
        
        # Add some context
        self.chat_history.add_exchange("What is Python?", "Python is a programming language.")
        self.chat_history.add_exchange("How do I install it?", "You can download Python from python.org")
        
        # Enhanced query should include context
        enhanced = self.chat_history.enhance_query_with_context(query)
        self.assertIn("conversation context", enhanced)
        self.assertIn("What is Python?", enhanced)
        self.assertIn(query, enhanced)
    
    def test_get_related_queries(self):
        """Test finding related queries."""
        # Add some exchanges
        self.chat_history.add_exchange("What is Python programming?", "Python is a language")
        self.chat_history.add_exchange("How to cook pasta?", "Boil water first")
        self.chat_history.add_exchange("Python data structures", "Lists, tuples, dicts")
        
        # Find queries related to Python
        related = self.chat_history.get_related_queries("Python libraries", similarity_threshold=0.1)
        
        # Should find Python-related queries but not cooking
        python_related = [q for q in related if "python" in q.lower()]
        self.assertGreater(len(python_related), 0)
        self.assertNotIn("How to cook pasta?", related)
    
    def test_conversation_summary(self):
        """Test getting conversation summary."""
        # Empty history
        summary = self.chat_history.get_conversation_summary()
        self.assertEqual(summary['total_exchanges'], 0)
        
        # Add some exchanges
        self.chat_history.add_exchange("What is Python?", "Python is a programming language")
        self.chat_history.add_exchange("How to learn Python?", "Start with basics")
        
        summary = self.chat_history.get_conversation_summary()
        self.assertEqual(summary['total_exchanges'], 2)
        self.assertEqual(summary['session_id'], "test_session")
        self.assertIn('topics', summary)
        self.assertIn('duration', summary)
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        # Add some exchanges
        self.chat_history.add_exchange("Question 1", "Answer 1")
        self.chat_history.add_exchange("Question 2", "Answer 2")
        
        self.assertEqual(len(self.chat_history.history), 2)
        
        # Clear history
        self.chat_history.clear_history()
        self.assertEqual(len(self.chat_history.history), 0)
    
    def test_export_history(self):
        """Test exporting conversation history."""
        # Add some exchanges
        self.chat_history.add_exchange("Question 1", "Answer 1")
        self.chat_history.add_exchange("Question 2", "Answer 2")
        
        # Test JSON export
        json_export = self.chat_history.export_history("json")
        self.assertIn("test_session", json_export)
        self.assertIn("Question 1", json_export)
        
        # Test Markdown export
        md_export = self.chat_history.export_history("markdown")
        self.assertIn("# Conversation History", md_export)
        self.assertIn("**User:**", md_export)
        self.assertIn("**Assistant:**", md_export)
    
    def test_persistence(self):
        """Test that history is saved and loaded correctly."""
        # Add an exchange
        self.chat_history.add_exchange("Test question", "Test answer")
        
        # Create a new ChatHistory instance with the same session
        new_chat = ChatHistory(
            session_id="test_session",
            history_dir=self.temp_dir
        )
        
        # Should load the previous history
        self.assertEqual(len(new_chat.history), 1)
        self.assertEqual(new_chat.history[0]['user_query'], "Test question")
    
    def test_factory_function(self):
        """Test the create_chat_history factory function."""
        chat = create_chat_history(session_id="factory_test", max_history=3)
        self.assertIsInstance(chat, ChatHistory)
        self.assertEqual(chat.session_id, "factory_test")
        self.assertEqual(chat.max_history, 3)
    
    def test_automatic_session_id(self):
        """Test automatic session ID generation."""
        chat = ChatHistory(history_dir=self.temp_dir)
        self.assertTrue(chat.session_id.startswith("chat_"))
        self.assertTrue(chat.session_id.replace("chat_", "").isdigit())


if __name__ == "__main__":
    unittest.main()
