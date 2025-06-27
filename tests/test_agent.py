"""
Tests for the reasoning agent
"""
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from nterm import ReasoningAgent, create_nterm


class TestReasoningAgent(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False)
        self.temp_db.close()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    @patch('nterm.agent.Agent')
    def test_agent_initialization(self, mock_agent):
        """Test that agent initializes correctly."""
        agent = ReasoningAgent(db_file=self.temp_db.name)
        
        self.assertIsNotNone(agent.agent)
        self.assertEqual(agent.model_id, "gpt-4.1")
        self.assertEqual(agent.db_file, self.temp_db.name)
        mock_agent.assert_called_once()
    
    @patch('nterm.agent.Agent')
    def test_custom_model_id(self, mock_agent):
        """Test agent with custom model ID."""
        custom_model = "gpt-4"
        agent = ReasoningAgent(model_id=custom_model, db_file=self.temp_db.name)
        
        self.assertEqual(agent.model_id, custom_model)
    
    @patch('nterm.agent.Agent')
    def test_query_method(self, mock_agent):
        """Test the query method."""
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_agent.return_value.run.return_value = mock_response
        
        agent = ReasoningAgent(db_file=self.temp_db.name)
        response = agent.query("Test query")
        
        self.assertEqual(response, "Test response")
        mock_agent.return_value.run.assert_called_once_with("Test query")
    
    @patch('nterm.agent.Agent')
    def test_factory_function(self, mock_agent):
        """Test the create_nterm factory function."""
        agent = create_nterm(db_file=self.temp_db.name)
        
        self.assertIsInstance(agent, ReasoningAgent)
        mock_agent.assert_called_once()
    
    @patch('nterm.agent.Agent')
    def test_add_tool(self, mock_agent):
        """Test adding custom tools."""
        mock_tool = Mock()
        mock_agent.return_value.tools = []
        
        agent = ReasoningAgent(db_file=self.temp_db.name)
        agent.add_tool(mock_tool)
        
        self.assertIn(mock_tool, agent.agent.tools)


if __name__ == '__main__':
    unittest.main()