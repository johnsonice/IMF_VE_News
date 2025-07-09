# General LLM Factory

A comprehensive, unified interface for multiple LLM providers with structured outputs using the Instructor library.

## Overview

The General LLM Factory provides a consistent API for working with different LLM providers including OpenAI, Google Gemini, Anthropic Claude, and any OpenAI-compatible APIs. Built on top of the Instructor library, it ensures reliable structured outputs using Pydantic models.

## Features

### 🔄 Multi-Provider Support
- **OpenAI**: GPT-4, GPT-4o, GPT-3.5-turbo, etc.
- **Google Gemini**: Gemini 1.5 Flash, Gemini 1.5 Pro
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- **OpenAI-Compatible**: Groq, Together AI, Anyscale, etc.
- **Azure OpenAI**: Enterprise deployment support

### 📊 Structured Outputs
- **Pydantic Integration**: Type-safe responses with automatic validation
- **Complex Schemas**: Support for nested objects, lists, and custom validators
- **JSON Mode**: Fallback to JSON parsing when needed
- **Streaming Support**: Real-time structured output streaming

### 🔧 Advanced Features
- **Async Support**: Full async/await compatibility
- **Retry Logic**: Automatic retry with exponential backoff
- **Error Handling**: Comprehensive error handling and logging
- **Provider Fallback**: Automatic fallback to alternative providers
- **Connection Testing**: Built-in connection testing utilities

## Installation

```bash
# Install required dependencies
pip install instructor openai

# Optional provider-specific dependencies
pip install google-generativeai  # For Google Gemini
pip install anthropic            # For Anthropic Claude
```

## Quick Start

### Basic Usage

```python
from llm_factory_general import create_openai_factory
from pydantic import BaseModel, Field

# Create a factory
factory = create_openai_factory()

# Simple text generation
messages = [{"role": "user", "content": "Hello, world!"}]
response = factory.get_response_content(messages)
print(response)

# Structured output
class User(BaseModel):
    name: str = Field(..., description="User's name")
    age: int = Field(..., description="User's age")

user = factory.get_structured_response(
    messages=[{"role": "user", "content": "John is 25 years old"}],
    response_model=User
)
print(f"Name: {user.name}, Age: {user.age}")
```

### Multiple Providers

```python
from llm_factory_general import (
    create_openai_factory,
    create_google_gemini_factory,
    create_anthropic_factory,
    create_openai_compatible_factory
)

# OpenAI
openai_factory = create_openai_factory(model_name="gpt-4")

# Google Gemini
gemini_factory = create_google_gemini_factory(model_name="gemini-1.5-flash")

# Anthropic
anthropic_factory = create_anthropic_factory(model_name="claude-3-5-sonnet-20241022")

# OpenAI-Compatible (Groq)
groq_factory = create_openai_compatible_factory(
    model_name="mixtral-8x7b-32768",
    base_url="https://api.groq.com/openai/v1",
    api_key="your-groq-api-key"
)
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Optional: OpenAI-Compatible APIs
GROQ_API_KEY=your_groq_api_key
```

### Advanced Configuration

```python
from llm_factory_general import GeneralLLMFactory, ModelConfig, ProviderType

# Custom configuration
config = ModelConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

factory = GeneralLLMFactory(config)
```

## Usage Examples

### Document Analysis

```python
from pydantic import BaseModel, Field
from typing import List

class DocumentSummary(BaseModel):
    title: str = Field(..., description="Document title")
    main_topics: List[str] = Field(..., description="Main topics covered")
    key_points: List[str] = Field(..., description="Key points")
    sentiment: str = Field(..., description="Overall sentiment")
    word_count: int = Field(..., description="Approximate word count")

factory = create_openai_factory()

messages = [
    {
        "role": "system",
        "content": "You are a document analysis expert."
    },
    {
        "role": "user",
        "content": f"Analyze this document: {document_text}"
    }
]

summary = factory.get_structured_response(messages, DocumentSummary)
print(f"Title: {summary.title}")
print(f"Topics: {summary.main_topics}")
```

### Async Processing

```python
import asyncio

async def async_example():
    factory = create_openai_factory()
    
    messages = [{"role": "user", "content": "Hello, async world!"}]
    response = await factory.get_response_content_async(messages)
    print(response)

    # Structured async response
    user = await factory.get_structured_response_async(
        messages=[{"role": "user", "content": "Alice is 30 years old"}],
        response_model=User
    )
    print(f"Name: {user.name}, Age: {user.age}")

asyncio.run(async_example())
```

### Provider Fallback Strategy

```python
class LLMManager:
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        try:
            self.providers['openai'] = create_openai_factory()
        except Exception as e:
            print(f"OpenAI failed: {e}")
        
        try:
            self.providers['anthropic'] = create_anthropic_factory()
        except Exception as e:
            print(f"Anthropic failed: {e}")
    
    def get_provider(self, preferred='openai'):
        if preferred in self.providers:
            return self.providers[preferred]
        
        # Fallback to any available provider
        if self.providers:
            return next(iter(self.providers.values()))
        
        raise ValueError("No providers available")

# Usage
manager = LLMManager()
factory = manager.get_provider('openai')  # Falls back if OpenAI unavailable
```

### Batch Processing

```python
async def batch_process_texts(texts: List[str]):
    factory = create_openai_factory()
    
    async def process_single(text: str):
        messages = [{"role": "user", "content": f"Analyze sentiment: {text}"}]
        return await factory.get_response_content_async(messages)
    
    # Process all texts concurrently
    results = await asyncio.gather(
        *[process_single(text) for text in texts]
    )
    
    return results

texts = ["I love this!", "This is terrible.", "It's okay."]
results = asyncio.run(batch_process_texts(texts))
```

## Error Handling

```python
from llm_factory_general import GeneralLLMFactory, ModelConfig, ProviderType

try:
    factory = create_openai_factory()
    response = factory.get_response_content(messages)
except Exception as e:
    print(f"Error: {e}")
    
    # Try alternative provider
    try:
        factory = create_anthropic_factory()
        response = factory.get_response_content(messages)
    except Exception as e:
        print(f"Fallback also failed: {e}")
```

## Testing

### Connection Tests

```python
# Test connection
factory = create_openai_factory()
try:
    result = factory.test_connection()
    print(f"Connection successful: {result}")
except Exception as e:
    print(f"Connection failed: {e}")

# Async connection test
result = await factory.test_connection_async()
```

### Running Tests

```bash
# Run the comprehensive test suite
cd test/
python test_general_llm_factory.py

# Run practical examples
cd examples/
python llm_factory_examples.py
```

## Provider-Specific Notes

### OpenAI
- Supports all GPT models including GPT-4, GPT-4o, GPT-3.5-turbo
- Function calling and tool use available
- Streaming support for real-time responses

### Google Gemini
- Supports Gemini 1.5 Flash and Pro models
- Multimodal capabilities (text, images, audio)
- Efficient for simple tasks with Flash model

### Anthropic Claude
- Supports Claude 3.5 Sonnet, Claude 3 Haiku, Claude 3 Opus
- Excellent for complex reasoning and analysis
- Large context window support

### OpenAI-Compatible APIs
- Works with Groq, Together AI, Anyscale, etc.
- Requires base URL and API key configuration
- Often provides faster inference speeds

## Performance Considerations

### Optimization Tips
1. **Model Selection**: Choose appropriate model size for your task
2. **Batch Processing**: Use async processing for multiple requests
3. **Caching**: Cache responses for repeated queries
4. **Streaming**: Use streaming for long responses
5. **Provider Selection**: Choose providers based on speed/cost requirements

### Rate Limiting
Each provider has different rate limits:
- **OpenAI**: Depends on tier and model
- **Google Gemini**: 15 requests/minute for free tier
- **Anthropic**: Varies by plan
- **OpenAI-Compatible**: Provider-specific limits

## Best Practices

1. **Environment Variables**: Store API keys securely
2. **Error Handling**: Implement comprehensive error handling
3. **Logging**: Enable logging for debugging
4. **Testing**: Test with multiple providers
5. **Validation**: Always validate structured outputs
6. **Fallback**: Implement provider fallback strategies

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install instructor openai google-generativeai anthropic
   ```

2. **API Key Issues**
   ```python
   # Check if API key is set
   import os
   print(os.getenv('OPENAI_API_KEY'))
   ```

3. **Rate Limiting**
   ```python
   # Implement exponential backoff (already included)
   from tenacity import retry, stop_after_attempt, wait_exponential
   ```

4. **Model Compatibility**
   ```python
   # Check available models
   factory = create_openai_factory()
   info = factory.get_provider_info()
   print(info)
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite to verify setup
3. Review the examples for usage patterns
4. Check provider-specific documentation

---

**Note**: This implementation is based on the Instructor library and provides a unified interface for multiple LLM providers. Make sure to check the latest documentation for each provider for any API changes. 