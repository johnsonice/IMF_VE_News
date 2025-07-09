"""
General LLM Factory for multiple providers using Instructor library.
Supports OpenAI, Google Gemini, Anthropic, and OpenAI-compatible APIs.
Provides unified API with structured outputs using Pydantic models.
"""

import os
import json
import re
from typing import Dict, List, Optional, Union, Any, Type
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import datetime
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

# Setup logging
now = datetime.datetime.now()
USER = os.environ.get("USER", "UNKNOWN").upper()
today = datetime.date.today()
file_path = f"log/{USER}/{today}"
os.makedirs(file_path, exist_ok=True)
filename = f"{file_path}/LLM-Factory-{now.hour:02d}:{now.minute:02d}.log"
fmt = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename=filename,
    filemode="w",
    format=fmt
)
logger = logging.getLogger(__name__)

# Import instructor and provider-specific clients
try:
    import instructor
    from openai import OpenAI, AsyncOpenAI
    from openai import AzureOpenAI, AsyncAzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google Generative AI not available")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available")

# Constants
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 30


class ProviderType(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GOOGLE_GEMINI = "google_gemini"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class ModelConfig(BaseModel):
    """Configuration for LLM models."""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT
    # Provider-specific configs
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = None
    google_project_id: Optional[str] = None
    anthropic_region: Optional[str] = None


class GeneralLLMFactory:
    """
    Universal LLM Factory supporting multiple providers with structured outputs.
    
    Supports:
    - OpenAI GPT models
    - Azure OpenAI
    - Google Gemini
    - Anthropic Claude
    - OpenAI-compatible APIs (Groq, Together, etc.)
    
    Features:
    - Unified API across all providers
    - Structured outputs using Pydantic models
    - Automatic retry logic
    - Async support
    - Proper error handling
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the factory with a model configuration.
        
        Args:
            config: ModelConfig object with provider settings
        """
        self.config = config
        self.client = None
        self.async_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        provider = self.config.provider
        
        if provider == ProviderType.OPENAI:
            self._init_openai_client()
        elif provider == ProviderType.AZURE_OPENAI:
            self._init_azure_openai_client()
        elif provider == ProviderType.GOOGLE_GEMINI:
            self._init_google_client()
        elif provider == ProviderType.ANTHROPIC:
            self._init_anthropic_client()
        elif provider == ProviderType.OPENAI_COMPATIBLE:
            self._init_openai_compatible_client()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        api_key = self.config.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = instructor.from_openai(
            OpenAI(api_key=api_key, timeout=self.config.timeout),
            mode=instructor.Mode.TOOLS
        )
        
        self.async_client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key, timeout=self.config.timeout),
            mode=instructor.Mode.TOOLS
        )
    
    def _init_azure_openai_client(self):
        """Initialize Azure OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        api_key = self.config.api_key or os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = self.config.azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = self.config.azure_api_version or "2024-02-01"
        
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI API key and endpoint required")
        
        self.client = instructor.from_openai(
            AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                timeout=self.config.timeout
            ),
            mode=instructor.Mode.TOOLS
        )
        
        self.async_client = instructor.from_openai(
            AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                timeout=self.config.timeout
            ),
            mode=instructor.Mode.TOOLS
        )
    
    def _init_google_client(self):
        """Initialize Google Gemini client."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI package not available. Install with: pip install google-generativeai")
        
        api_key = self.config.api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key required")
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Use instructor's Google integration
        self.client = instructor.from_gemini(
            genai.GenerativeModel(self.config.model_name,
                                  generation_config={
                                      'temperature': self.config.temperature,
                                      'max_output_tokens': self.config.max_tokens,
                                  }),
            mode=instructor.Mode.GEMINI_JSON
        )
        
        # Note: Google doesn't have official async client, using sync for now
        self.async_client = self.client
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not available. Install with: pip install anthropic")
        
        api_key = self.config.api_key or os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        self.client = instructor.from_anthropic(
            anthropic.Anthropic(api_key=api_key, timeout=self.config.timeout),
            mode=instructor.Mode.ANTHROPIC_JSON
        )
        
        self.async_client = instructor.from_anthropic(
            anthropic.AsyncAnthropic(api_key=api_key, timeout=self.config.timeout),
            mode=instructor.Mode.ANTHROPIC_JSON
        )
    
    def _init_openai_compatible_client(self):
        """Initialize OpenAI-compatible client (Groq, Together, etc.)."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        api_key = self.config.api_key
        base_url = self.config.base_url
        
        if not api_key or not base_url:
            raise ValueError("API key and base URL required for OpenAI-compatible provider")
        
        self.client = instructor.from_openai(
            OpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout),
            mode=instructor.Mode.TOOLS
        )
        
        self.async_client = instructor.from_openai(
            AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=self.config.timeout),
            mode=instructor.Mode.TOOLS
        )
    
    @staticmethod
    def _is_pydantic_model(obj) -> bool:
        """Check if an object is a Pydantic model class."""
        return (obj and hasattr(obj, '__bases__') and 
                any(base.__name__ == 'BaseModel' for base in obj.__mro__))
    
    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare messages for the specific provider."""
        # Provider-specific message formatting can be added here
        return messages
    
    def _get_model_name(self) -> str:
        """Get the model name for the API call."""
        return self.config.model_name
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _make_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Make a retry-enabled API call with provider-specific handling."""
        prepared_messages = self._prepare_messages(messages)
        
        # Ensure response_model is always present - default to str if not provided
        response_model = kwargs.get('response_model', str)
        
        call_kwargs = {
            'messages': prepared_messages,
            'model': self._get_model_name(),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'response_model': response_model,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'response_model']}
        }
        
        # Handle provider-specific API differences
        if self.config.provider == ProviderType.ANTHROPIC:
            call_kwargs['max_tokens'] = call_kwargs.pop('max_tokens', self.config.max_tokens)
        elif self.config.provider == ProviderType.GOOGLE_GEMINI:
            # Google uses different parameter names
            call_kwargs = {
                'messages': prepared_messages,
                'response_model': response_model,
            }
        #print(call_kwargs)
        return self.client.chat.completions.create(**call_kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _make_async_api_call(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Make an async retry-enabled API call."""
        prepared_messages = self._prepare_messages(messages)
        
        # Ensure response_model is always present - default to str if not provided
        response_model = kwargs.get('response_model', str)
        
        call_kwargs = {
            'messages': prepared_messages,
            'model': self._get_model_name(),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'response_model': response_model,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens', 'response_model']}
        }
        
        # Handle provider-specific API differences
        if self.config.provider == ProviderType.ANTHROPIC:
            call_kwargs['max_tokens'] = call_kwargs.pop('max_tokens', self.config.max_tokens)
        elif self.config.provider == ProviderType.GOOGLE_GEMINI:
            call_kwargs = {
                'messages': prepared_messages,
                'response_model': response_model,
            }
        
        return await self.async_client.chat.completions.create(**call_kwargs)
    
    def get_response_content(self, messages: List[Dict[str, str]], **kwargs) -> Union[str, Any]:
        """Get response content from the model."""
        try:
            response = self._make_api_call(messages, **kwargs)
            
            response_format = kwargs.get("response_model")
            if self._is_pydantic_model(response_format):
                return response  # Already parsed by instructor
            else:
                # Handle different response structures from instructor
                if hasattr(response, 'choices') and response.choices:
                    if hasattr(response.choices[0], 'message'):
                        return response.choices[0].message.content
                    else:
                        return str(response.choices[0])
                elif hasattr(response, 'content'):
                    return response.content
                elif hasattr(response, 'text'):
                    return response.text
                else:
                    # If it's a string or other simple type, return as is
                    return str(response)
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    async def get_response_content_async(self, messages: List[Dict[str, str]], **kwargs) -> Union[str, Any]:
        """Get response content from the model asynchronously."""
        try:
            response = await self._make_async_api_call(messages, **kwargs)
            
            response_format = kwargs.get("response_model")
            if self._is_pydantic_model(response_format):
                return response  # Already parsed by instructor
            else:
                # Handle different response structures from instructor
                if hasattr(response, 'choices') and response.choices:
                    if hasattr(response.choices[0], 'message'):
                        return response.choices[0].message.content
                    else:
                        return str(response.choices[0])
                elif hasattr(response, 'content'):
                    return response.content
                elif hasattr(response, 'text'):
                    return response.text
                else:
                    # If it's a string or other simple type, return as is
                    return str(response)
                
        except Exception as e:
            logger.error(f"Async API call failed: {e}")
            raise
    
    def get_structured_response(self, messages: List[Dict[str, str]], response_model: Type[BaseModel], **kwargs) -> BaseModel:
        """Get a structured response using a Pydantic model."""
        if not self._is_pydantic_model(response_model):
            raise ValueError("response_model must be a Pydantic BaseModel class")
        
        kwargs["response_model"] = response_model
        return self.get_response_content(messages, **kwargs)
    
    async def get_structured_response_async(self, messages: List[Dict[str, str]], response_model: Type[BaseModel], **kwargs) -> BaseModel:
        """Get a structured response using a Pydantic model asynchronously."""
        if not self._is_pydantic_model(response_model):
            raise ValueError("response_model must be a Pydantic BaseModel class")
        
        kwargs["response_model"] = response_model
        return await self.get_response_content_async(messages, **kwargs)
    
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
    
    def test_connection(self, test_message: str = "Hello, can you respond with 'Connection successful'?") -> str:
        """Test the API connection."""
        try:
            messages = [{"role": "user", "content": test_message}]
            response = self.get_response_content(messages)
            logger.info(f"✅ Connection test successful for {self.config.provider}")
            return response
        except Exception as e:
            logger.error(f"❌ Connection test failed for {self.config.provider}: {e}")
            raise
    
    async def test_connection_async(self, test_message: str = "Hello, can you respond with 'Connection successful'?") -> str:
        """Test the API connection asynchronously."""
        try:
            messages = [{"role": "user", "content": test_message}]
            response = await self.get_response_content_async(messages)
            logger.info(f"✅ Async connection test successful for {self.config.provider}")
            return response
        except Exception as e:
            logger.error(f"❌ Async connection test failed for {self.config.provider}: {e}")
            raise
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider and model."""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "base_url": self.config.base_url,
            "supports_async": self.async_client is not None
        }


# Factory functions for easy instantiation
def create_openai_factory(
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> GeneralLLMFactory:
    """Create an OpenAI LLM factory."""
    config = ModelConfig(
        provider=ProviderType.OPENAI,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return GeneralLLMFactory(config)


def create_google_gemini_factory(
    model_name: str = "gemini-1.5-flash",
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> GeneralLLMFactory:
    """Create a Google Gemini LLM factory."""
    config = ModelConfig(
        provider=ProviderType.GOOGLE_GEMINI,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return GeneralLLMFactory(config)


def create_anthropic_factory(
    model_name: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> GeneralLLMFactory:
    """Create an Anthropic LLM factory."""
    config = ModelConfig(
        provider=ProviderType.ANTHROPIC,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return GeneralLLMFactory(config)


def create_openai_compatible_factory(
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> GeneralLLMFactory:
    """Create an OpenAI-compatible LLM factory (Groq, Together, etc.)."""
    config = ModelConfig(
        provider=ProviderType.OPENAI_COMPATIBLE,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return GeneralLLMFactory(config)


# Example usage
if __name__ == "__main__":
    # Example usage with different providers
    print("=== General LLM Factory Examples ===")
    
    # OpenAI example
    try:
        openai_factory = create_openai_factory()
        print("OpenAI factory created successfully")
        
        # Test connection
        result = openai_factory.test_connection()
        print(f"OpenAI test result: {result}")
        
        # Get provider info
        info = openai_factory.get_provider_info()
        print(f"OpenAI provider info: {info}")
        
    except Exception as e:
        print(f"OpenAI factory failed: {e}")
    
    # You can uncomment these as needed:
    
    # # Google Gemini example
    # try:
    #     gemini_factory = create_google_gemini_factory()
    #     print("Google Gemini factory created successfully")
    #     result = gemini_factory.test_connection()
    #     print(f"Gemini test result: {result}")
    # except Exception as e:
    #     print(f"Gemini factory failed: {e}")
    
    # # Anthropic example
    # try:
    #     anthropic_factory = create_anthropic_factory()
    #     print("Anthropic factory created successfully")
    #     result = anthropic_factory.test_connection()
    #     print(f"Anthropic test result: {result}")
    # except Exception as e:
    #     print(f"Anthropic factory failed: {e}")
    
    # # OpenAI-compatible example (Groq)
    # try:
    #     groq_factory = create_openai_compatible_factory(
    #         model_name="mixtral-8x7b-32768",
    #         base_url="https://api.groq.com/openai/v1",
    #         api_key=os.getenv("GROQ_API_KEY")
    #     )
    #     print("Groq factory created successfully")
    #     result = groq_factory.test_connection()
    #     print(f"Groq test result: {result}")
    # except Exception as e:
    #     print(f"Groq factory failed: {e}")
