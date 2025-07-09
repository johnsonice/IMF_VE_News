"""
Comprehensive test suite for General LLM Factory.

This test suite demonstrates:
1. Multiple provider support (OpenAI, Google Gemini, Anthropic, OpenAI-compatible)
2. Structured outputs with Pydantic models
3. Async functionality
4. Error handling and connection testing
5. Provider-specific features

Setup Instructions:
1. Create a .env file in the IMF_VE_News directory (one level up from Factiva_News)
2. Add your API keys:
   - OPENAI_API_KEY=your_openai_key
   - GOOGLE_API_KEY=your_google_key
   - ANTHROPIC_API_KEY=your_anthropic_key
   - GROQ_API_KEY=your_groq_key (optional)
3. Install dependencies: pip install instructor google-generativeai anthropic

"""

import sys
import os
import asyncio
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add the libs directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(os.path.dirname(current_dir), 'libs')
if libs_dir not in sys.path:
    sys.path.insert(0, libs_dir)

# Load environment variables
load_dotenv('../../.env')

# Import our general LLM factory
from llm_factory_general import (
    GeneralLLMFactory,
    ModelConfig,
    ProviderType,
    create_openai_factory,
    create_google_gemini_factory,
    create_anthropic_factory,
    create_openai_compatible_factory
)

# Test Pydantic models
class Capital(BaseModel):
    """Model for capital city information."""
    city: str = Field(..., description="Name of the capital city")
    country: str = Field(..., description="Name of the country")
    population: int = Field(..., description="Population count")
    landmarks: List[str] = Field(default_factory=list, description="Notable landmarks")

class CapitalsResponse(BaseModel):
    """Response model for multiple capitals."""
    capitals: List[Capital] = Field(default_factory=list)

class SimpleMessage(BaseModel):
    """Simple message model for testing."""
    message: str = Field(..., description="A simple message")
    confidence: float = Field(..., description="Confidence level between 0 and 1")

class CodeAnalysis(BaseModel):
    """Model for code analysis."""
    language: str = Field(..., description="Programming language")
    complexity: str = Field(..., description="Code complexity level")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    estimated_time: int = Field(..., description="Estimated time to complete in minutes")


def test_provider_creation():
    """Test factory creation for different providers."""
    print("=== Testing Provider Creation ===")
    
    # Test OpenAI factory
    try:
        openai_factory = create_openai_factory()
        print("✅ OpenAI factory created successfully")
        info = openai_factory.get_provider_info()
        print(f"   Provider info: {info}")
    except Exception as e:
        print(f"❌ OpenAI factory failed: {e}")
    
    # Test Google Gemini factory
    try:
        gemini_factory = create_google_gemini_factory()
        print("✅ Google Gemini factory created successfully")
        info = gemini_factory.get_provider_info()
        print(f"   Provider info: {info}")
    except Exception as e:
        print(f"❌ Google Gemini factory failed: {e}")
    
    # Test Anthropic factory
    try:
        anthropic_factory = create_anthropic_factory()
        print("✅ Anthropic factory created successfully")
        info = anthropic_factory.get_provider_info()
        print(f"   Provider info: {info}")
    except Exception as e:
        print(f"❌ Anthropic factory failed: {e}")
    
    # # Test OpenAI-compatible factory (Groq)
    # try:
    #     groq_api_key = os.getenv("GROQ_API_KEY")
    #     if groq_api_key:
    #         groq_factory = create_openai_compatible_factory(
    #             model_name="mixtral-8x7b-32768",
    #             base_url="https://api.groq.com/openai/v1",
    #             api_key=groq_api_key
    #         )
    #         print("✅ Groq factory created successfully")
    #         info = groq_factory.get_provider_info()
    #         print(f"   Provider info: {info}")
    #     else:
    #         print("⚠️  Groq API key not found, skipping Groq test")
    # except Exception as e:
    #     print(f"❌ Groq factory failed: {e}")


def test_basic_text_generation():
    """Test basic text generation across providers."""
    print("\n=== Testing Basic Text Generation ===")
    
    factories = []
    
    # OpenAI
    try:
        factories.append(("OpenAI", create_openai_factory()))
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
    
    # Google Gemini
    try:
        factories.append(("Google Gemini", create_google_gemini_factory()))
    except Exception as e:
        print(f"❌ Google Gemini setup failed: {e}")
    
    # Anthropic
    try:
        factories.append(("Anthropic", create_anthropic_factory()))
    except Exception as e:
        print(f"❌ Anthropic setup failed: {e}")
    
    test_messages = [
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
    ]
    
    for provider_name, factory in factories:
        try:
            response = factory.get_response_content(test_messages)
            print(f"✅ {provider_name}: {response}")
        except Exception as e:
            print(f"❌ {provider_name} failed: {e}")


def test_structured_outputs():
    """Test structured outputs with Pydantic models."""
    print("\n=== Testing Structured Outputs ===")
    
    factories = []
    
    # OpenAI
    try:
        factories.append(("OpenAI", create_openai_factory()))
    except Exception as e:
        print(f"❌ OpenAI setup failed: {e}")
    
    # Google Gemini
    try:
        factories.append(("Google Gemini", create_google_gemini_factory()))
    except Exception as e:
        print(f"❌ Google Gemini setup failed: {e}")
    
    # Anthropic
    try:
        factories.append(("Anthropic", create_anthropic_factory()))
    except Exception as e:
        print(f"❌ Anthropic setup failed: {e}")
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides structured data about European capitals."
        },
        {
            "role": "user",
            "content": "List 2 European capitals with their details including city, country, population, and notable landmarks."
        }
    ]
    
    for provider_name, factory in factories:
        try:
            print(f"\n--- Testing {provider_name} structured output ---")
            
            # Test with Pydantic model
            capitals_response = factory.get_structured_response(messages, CapitalsResponse)
            print(f"✅ {provider_name} structured response successful:")
            for capital in capitals_response.capitals:
                print(f"  {capital.city}, {capital.country}")
                print(f"  Population: {capital.population:,}")
                print(f"  Landmarks: {', '.join(capital.landmarks)}")
                print()
            
        except Exception as e:
            print(f"❌ {provider_name} structured output failed: {e}")


def test_code_analysis():
    """Test code analysis with structured output."""
    print("\n=== Testing Code Analysis ===")
    
    try:
        factory = create_openai_factory()
        
        messages = [
            {
                "role": "user",
                "content": """
                Analyze this Python code:
                
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                
                Provide analysis including language, complexity, suggestions, and estimated time to optimize.
                """
            }
        ]
        
        analysis = factory.get_structured_response(messages, CodeAnalysis)
        print("✅ Code analysis successful:")
        print(f"  Language: {analysis.language}")
        print(f"  Complexity: {analysis.complexity}")
        print(f"  Suggestions: {analysis.suggestions}")
        print(f"  Estimated time: {analysis.estimated_time} minutes")
        
    except Exception as e:
        print(f"❌ Code analysis failed: {e}")


async def test_async_functionality():
    """Test asynchronous functionality."""
    print("\n=== Testing Async Functionality ===")
    
    try:
        factory = create_openai_factory()
        
        messages = [
            {"role": "user", "content": "Generate a simple greeting message with confidence level."}
        ]
        
        # Test async structured response
        response = await factory.get_structured_response_async(messages, SimpleMessage)
        print("✅ Async structured response successful:")
        print(f"  Message: {response.message}")
        print(f"  Confidence: {response.confidence}")
        
        # Test async text response
        text_messages = [
            {"role": "user", "content": "What is machine learning in one sentence?"}
        ]
        
        text_response = await factory.get_response_content_async(text_messages)
        print(f"✅ Async text response: {text_response}")
        
    except Exception as e:
        print(f"❌ Async functionality failed: {e}")


def test_provider_specific_models():
    """Test provider-specific model configurations."""
    print("\n=== Testing Provider-Specific Models ===")
    
    # Test different OpenAI models
    try:
        gpt4_factory = create_openai_factory(model_name="gpt-4")
        gpt3_factory = create_openai_factory(model_name="gpt-3.5-turbo")
        
        test_messages = [
            {"role": "user", "content": "Hello, which model are you?"}
        ]
        
        print("✅ Multiple OpenAI models:")
        gpt4_response = gpt4_factory.get_response_content(test_messages)
        print(f"  GPT-4: {gpt4_response[:100]}...")
        
        gpt3_response = gpt3_factory.get_response_content(test_messages)
        print(f"  GPT-3.5: {gpt3_response[:100]}...")
        
    except Exception as e:
        print(f"❌ Provider-specific models failed: {e}")


def test_error_handling():
    """Test error handling and retry logic."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with invalid API key
        config = ModelConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4o-mini",
            api_key="invalid_key"
        )
        
        factory = GeneralLLMFactory(config)
        
        messages = [
            {"role": "user", "content": "This should fail"}
        ]
        
        response = factory.get_response_content(messages)
        print("❌ Error handling test failed - should have raised exception")
        
    except Exception as e:
        print(f"✅ Error handling working correctly: {type(e).__name__}")


def test_connection_tests():
    """Test connection testing functionality."""
    print("\n=== Testing Connection Tests ===")
    
    # Test OpenAI connection
    try:
        factory = create_openai_factory()
        result = factory.test_connection()
        print(f"✅ OpenAI connection test: {result}")
    except Exception as e:
        print(f"❌ OpenAI connection test failed: {e}")
    
    # Test async connection
    try:
        factory = create_openai_factory()
        result = asyncio.run(factory.test_connection_async())
        print(f"✅ OpenAI async connection test: {result}")
    except Exception as e:
        print(f"❌ OpenAI async connection test failed: {e}")


def main():
    """Run all tests."""
    print("=== General LLM Factory Test Suite ===")
    print("This test suite demonstrates multi-provider LLM functionality with structured outputs.\n")
    
    # Run all tests
    test_provider_creation()
    test_basic_text_generation()
    test_structured_outputs()
    test_code_analysis()
    test_provider_specific_models()
    test_error_handling()
    test_connection_tests()
    
    # Run async tests
    print("\n=== Running Async Tests ===")
    asyncio.run(test_async_functionality())
    
    print("\n=== Test Suite Complete ===")
    print("Note: Some tests may fail if API keys are not configured or providers are not available.")


if __name__ == "__main__":
    main() 