"""
Tests for the reasoning agent
"""
import unittest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from nterm import ElegantReasoningAgent, create_elegant_nterm


class TestElegantReasoningAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        
        # Mock environment variables
        self.original_env = os.environ.copy()
        os.environ['OPENAI_API_KEY'] = 'test-api-key'
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    @patch('nterm.enhancedagent.Agent')
    @patch('nterm.enhancedagent.Console')
    def test_agent_initialization(self, mock_console, mock_agent):
        """Test that agent initializes correctly."""
        agent = ElegantReasoningAgent(db_file=self.temp_db.name)
        
        self.assertIsNotNone(agent.agent)
        self.assertEqual(agent.db_file, self.temp_db.name)
        self.assertTrue(hasattr(agent, 'console'))
        mock_agent.assert_called_once()
    
    @patch('nterm.enhancedagent.Agent')
    @patch('nterm.enhancedagent.Console')
    def test_custom_model_id(self, mock_console, mock_agent):
        """Test agent with custom model ID."""
        custom_model = "gpt-4"
        agent = ElegantReasoningAgent(model_id=custom_model, db_file=self.temp_db.name)
        
        self.assertEqual(agent.model_id, custom_model)
    
    @patch('nterm.enhancedagent.Agent')
    @patch('nterm.enhancedagent.Console')
    def test_query_method(self, mock_console, mock_agent):
        """Test the query method."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.content = "Test response"
        mock_agent.return_value.run.return_value = mock_response
        
        agent = ElegantReasoningAgent(db_file=self.temp_db.name)
        response = agent.query("Test query")
        
        self.assertEqual(response, "Test response")
        mock_agent.return_value.run.assert_called_once_with("Test query", stream=False)
    
    @patch('nterm.enhancedagent.Agent')
    @patch('nterm.enhancedagent.Console')
    def test_factory_function(self, mock_console, mock_agent):
        """Test the create_elegant_nterm factory function."""
        agent = create_elegant_nterm(db_file=self.temp_db.name)
        
        self.assertIsInstance(agent, ElegantReasoningAgent)
        mock_agent.assert_called_once()
    
    @patch('nterm.enhancedagent.Agent')
    @patch('nterm.enhancedagent.Console')
    def test_status_method(self, mock_console, mock_agent):
        """Test the get_status method."""
        agent = ElegantReasoningAgent(db_file=self.temp_db.name)
        status = agent.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('model_id', status)
        self.assertIn('db_file', status)


if __name__ == '__main__':
    unittest.main()