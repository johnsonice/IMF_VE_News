"""
Simple LLM Agent for OpenAI API interactions.
Provides basic API calls with retry logic and result parsing.
"""
#%%
import os
import json
import re
from typing import Dict, List, Optional, Union, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import datetime
now = datetime.datetime.now()
USER = os.environ.get("USER", "UNKNOWN").upper()
today = datetime.date.today()
file_path = f"log/{USER}/{today}"
os.makedirs(file_path, exist_ok=True)
filename = f"{file_path}/Exp-{now.hour:02d}:{now.minute:02d}.log"
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=filename,
    filemode="w",
    format=fmt
)
logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field
# Load API key from environment
from dotenv import load_dotenv
load_dotenv('../.env')

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0


#%%
class SimpleLLMAgent:
    """Simple LLM agent for OpenAI API interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize the agent with OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, uses environment variable OPENAI_API_KEY
            base_url: Base URL for API endpoint. If None, uses OpenAI default
            model: Model identifier to use (default: gpt-4o-mini)
            temperature: Temperature parameter for generation (default: 0)
        """
        self.client = OpenAI(
            api_key=api_key or os.environ.get('OPENAI_API_KEY'),
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
    
    @staticmethod
    def _is_pydantic_model(obj) -> bool:
        """Check if an object is a Pydantic model class."""
        return (obj and hasattr(obj, '__bases__') and 
                any(base.__name__ == 'BaseModel' for base in obj.__mro__))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Make a retry-enabled API call to OpenAI with Pydantic model support."""
        response_format = kwargs.get("response_format")
        
        # Check if response_format is a Pydantic model
        if self._is_pydantic_model(response_format):
            # Use beta.chat.completions.parse for Pydantic models
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop("response_format", None)  # Remove from kwargs to avoid duplication
            
            return self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format,
                **kwargs_copy
            )
        else:
            # Use regular API for JSON objects and text responses
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                **kwargs
            )

    def get_response_content(self, messages: List[Dict[str, str]], **kwargs) -> Union[str, Any]:
        """Get just the content from a completion response."""
        response = self._make_api_call(messages, **kwargs)
        response_format = kwargs.get("response_format")
        
        # Check if response_format is a Pydantic model (structured output)
        if self._is_pydantic_model(response_format):
            # Return parsed Pydantic model
            return response.choices[0].message.parsed
        else:
            # Return text content for regular responses and JSON objects
            return response.choices[0].message.content
    
    def get_structured_response(self, messages: List[Dict[str, str]], response_model, **kwargs):
        """Get a structured response using a Pydantic model."""
        if not self._is_pydantic_model(response_model):
            raise ValueError("response_model must be a Pydantic BaseModel class")
        
        kwargs["response_format"] = response_model
        return self.get_response_content(messages, **kwargs)

    @staticmethod
    def parse_json(text: str) -> Dict:
        """Parse JSON from response, handling code blocks."""
        # Remove code block markers if present
        text = re.sub(r'```json\s*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Text: {text}")
            raise

    def test_connection(self, test_message: str = "Hello, can you respond?") -> str:
        """Test the API connection."""
        try:
            messages = [{"role": "user", "content": test_message}]
            response = self._make_api_call(messages)
            result = response.choices[0].message.content
            logger.info("✅ Connection test successful")
            return result
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            raise
        

#%%
# Example usage
if __name__ == "__main__":
    # Create agent
    agent = SimpleLLMAgent()
    
    # Test connection
    print("Testing connection...")
    result = agent.test_connection()
    print(f"Response: {result}")
    
    # Example direct API call
    print("\nExample direct API call:")
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = agent._make_api_call(messages)
    print(f"Response: {response.choices[0].message.content}")
    
    # Example using result parsing function
    print("\nExample using result parsing function:")
    messages = [{"role": "user", "content": "What is the capital of Japan?"}]
    content = agent.get_response_content(messages)
    print(f"Parsed content: {content}")
    
    # Example structured output test
    print("\n=== Structured Output Tests ===")
    print("Run 'python unit_test.py' to execute structured output tests")
#%%