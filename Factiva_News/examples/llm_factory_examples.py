"""
Practical Examples for General LLM Factory

This script demonstrates real-world usage patterns for the General LLM Factory
with different providers and use cases.

Examples included:
1. Document analysis across multiple providers
2. Code generation and review
3. Data extraction from unstructured text
4. Multilingual translation
5. Batch processing with different models
6. Error handling and fallback strategies
"""

import sys
import os
import asyncio
from typing import List, Dict, Any, Optional
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

# Example Pydantic models for different use cases
class DocumentSummary(BaseModel):
    """Document summary model."""
    title: str = Field(..., description="Document title")
    main_topics: List[str] = Field(..., description="Main topics covered")
    key_points: List[str] = Field(..., description="Key points from the document")
    sentiment: str = Field(..., description="Overall sentiment: positive, negative, or neutral")
    word_count: int = Field(..., description="Approximate word count")
    reading_time: int = Field(..., description="Estimated reading time in minutes")

class PersonInfo(BaseModel):
    """Person information extraction model."""
    name: str = Field(..., description="Person's full name")
    age: Optional[int] = Field(None, description="Person's age")
    occupation: Optional[str] = Field(None, description="Person's occupation")
    location: Optional[str] = Field(None, description="Person's location")
    achievements: List[str] = Field(default_factory=list, description="Notable achievements")

class ProductReview(BaseModel):
    """Product review analysis model."""
    product_name: str = Field(..., description="Name of the product")
    rating: int = Field(..., description="Rating out of 5")
    pros: List[str] = Field(..., description="Positive aspects")
    cons: List[str] = Field(..., description="Negative aspects")
    recommendation: str = Field(..., description="Purchase recommendation")
    target_audience: str = Field(..., description="Who should buy this product")

class CodeReview(BaseModel):
    """Code review model."""
    language: str = Field(..., description="Programming language")
    code_quality: str = Field(..., description="Overall code quality rating")
    issues: List[str] = Field(default_factory=list, description="Issues found")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    complexity_score: int = Field(..., description="Complexity score (1-10)")
    maintainability: str = Field(..., description="Maintainability assessment")

class MultilingualResponse(BaseModel):
    """Multilingual response model."""
    original_language: str = Field(..., description="Detected original language")
    translated_text: str = Field(..., description="Translated text")
    confidence: float = Field(..., description="Translation confidence (0-1)")
    cultural_notes: List[str] = Field(default_factory=list, description="Cultural context notes")


class LLMFactoryManager:
    """Manager class for handling multiple LLM providers with fallback strategies."""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers."""
        # Try to initialize OpenAI
        try:
            self.providers['openai'] = create_openai_factory()
            print("✅ OpenAI provider initialized")
        except Exception as e:
            print(f"❌ OpenAI provider failed: {e}")
        
        # Try to initialize Google Gemini
        try:
            self.providers['gemini'] = create_google_gemini_factory()
            print("✅ Google Gemini provider initialized")
        except Exception as e:
            print(f"❌ Google Gemini provider failed: {e}")
        
        # Try to initialize Anthropic
        try:
            self.providers['anthropic'] = create_anthropic_factory()
            print("✅ Anthropic provider initialized")
        except Exception as e:
            print(f"❌ Anthropic provider failed: {e}")
        
        # Try to initialize Groq (OpenAI-compatible)
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key:
                self.providers['groq'] = create_openai_compatible_factory(
                    model_name="mixtral-8x7b-32768",
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_api_key
                )
                print("✅ Groq provider initialized")
        except Exception as e:
            print(f"❌ Groq provider failed: {e}")
    
    def get_provider(self, preferred_provider: str = 'openai') -> GeneralLLMFactory:
        """Get a provider with fallback logic."""
        if preferred_provider in self.providers:
            return self.providers[preferred_provider]
        
        # Fallback to any available provider
        if self.providers:
            provider_name = next(iter(self.providers))
            print(f"⚠️ Falling back to {provider_name} provider")
            return self.providers[provider_name]
        
        raise ValueError("No providers available")
    
    def get_all_providers(self) -> Dict[str, GeneralLLMFactory]:
        """Get all available providers."""
        return self.providers


def example_document_analysis():
    """Example: Document analysis and summarization."""
    print("\n=== Document Analysis Example ===")
    
    manager = LLMFactoryManager()
    
    # Sample document
    document = """
    Artificial Intelligence (AI) has been transforming industries at an unprecedented pace. 
    From healthcare to finance, AI applications are revolutionizing how we work and live. 
    Machine learning algorithms can now diagnose diseases, predict market trends, and even 
    create art. However, this rapid advancement also raises important ethical questions 
    about privacy, job displacement, and the need for regulation. As we move forward, 
    it's crucial to balance innovation with responsible development to ensure AI benefits 
    all of humanity.
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a document analysis expert. Analyze the given document and provide a comprehensive summary."
        },
        {
            "role": "user",
            "content": f"Analyze this document and provide a summary:\n\n{document}"
        }
    ]
    
    try:
        factory = manager.get_provider('openai')
        summary = factory.get_structured_response(messages, DocumentSummary)
        
        print("✅ Document analysis successful:")
        print(f"  Title: {summary.title}")
        print(f"  Main topics: {', '.join(summary.main_topics)}")
        print(f"  Key points: {summary.key_points}")
        print(f"  Sentiment: {summary.sentiment}")
        print(f"  Word count: {summary.word_count}")
        print(f"  Reading time: {summary.reading_time} minutes")
        
    except Exception as e:
        print(f"❌ Document analysis failed: {e}")


def example_person_extraction():
    """Example: Extract person information from text."""
    print("\n=== Person Information Extraction Example ===")
    
    manager = LLMFactoryManager()
    
    # Sample text with person information
    text = """
    Dr. Sarah Johnson, a 45-year-old neuroscientist from Stanford University, has been 
    leading groundbreaking research in brain-computer interfaces. Born in San Francisco, 
    she completed her PhD at MIT and has published over 100 research papers. Her work 
    on neural prosthetics has helped thousands of paralyzed patients regain mobility. 
    She recently won the National Science Foundation's highest honor for her contributions 
    to neurotechnology.
    """
    
    messages = [
        {
            "role": "user",
            "content": f"Extract person information from this text:\n\n{text}"
        }
    ]
    
    try:
        factory = manager.get_provider('openai')
        person_info = factory.get_structured_response(messages, PersonInfo)
        
        print("✅ Person extraction successful:")
        print(f"  Name: {person_info.name}")
        print(f"  Age: {person_info.age}")
        print(f"  Occupation: {person_info.occupation}")
        print(f"  Location: {person_info.location}")
        print(f"  Achievements: {person_info.achievements}")
        
    except Exception as e:
        print(f"❌ Person extraction failed: {e}")


def example_product_review_analysis():
    """Example: Analyze product reviews."""
    print("\n=== Product Review Analysis Example ===")
    
    manager = LLMFactoryManager()
    
    # Sample product review
    review = """
    I've been using the UltraBook Pro 15 for three months now, and I'm really impressed. 
    The display is stunning with vibrant colors and sharp text. The keyboard feels great 
    for long typing sessions, and the battery easily lasts 8-10 hours. The build quality 
    is solid, and it's surprisingly light for a 15-inch laptop. However, the speakers 
    could be louder, and it gets warm during intensive tasks. The price is steep at $1,800, 
    but for professionals who need a reliable workstation, it's worth it. I'd definitely 
    recommend it to developers and content creators.
    """
    
    messages = [
        {
            "role": "user",
            "content": f"Analyze this product review and provide structured feedback:\n\n{review}"
        }
    ]
    
    try:
        factory = manager.get_provider('openai')
        review_analysis = factory.get_structured_response(messages, ProductReview)
        
        print("✅ Product review analysis successful:")
        print(f"  Product: {review_analysis.product_name}")
        print(f"  Rating: {review_analysis.rating}/5")
        print(f"  Pros: {review_analysis.pros}")
        print(f"  Cons: {review_analysis.cons}")
        print(f"  Recommendation: {review_analysis.recommendation}")
        print(f"  Target audience: {review_analysis.target_audience}")
        
    except Exception as e:
        print(f"❌ Product review analysis failed: {e}")


def example_code_review():
    """Example: Code review and analysis."""
    print("\n=== Code Review Example ===")
    
    manager = LLMFactoryManager()
    
    # Sample code to review
    code = """
    def calculate_factorial(n):
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result = result * i
        return result
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # Usage
    print(calculate_factorial(5))
    print(fibonacci(10))
    """
    
    messages = [
        {
            "role": "user",
            "content": f"Review this Python code and provide detailed feedback:\n\n{code}"
        }
    ]
    
    try:
        factory = manager.get_provider('openai')
        code_review = factory.get_structured_response(messages, CodeReview)
        
        print("✅ Code review successful:")
        print(f"  Language: {code_review.language}")
        print(f"  Code quality: {code_review.code_quality}")
        print(f"  Issues: {code_review.issues}")
        print(f"  Suggestions: {code_review.suggestions}")
        print(f"  Complexity score: {code_review.complexity_score}/10")
        print(f"  Maintainability: {code_review.maintainability}")
        
    except Exception as e:
        print(f"❌ Code review failed: {e}")


async def example_async_batch_processing():
    """Example: Async batch processing with multiple providers."""
    print("\n=== Async Batch Processing Example ===")
    
    manager = LLMFactoryManager()
    
    # Sample texts to process
    texts = [
        "The weather is beautiful today!",
        "I'm feeling a bit sad about the news.",
        "This is absolutely amazing! I love it!",
        "The product quality is disappointing."
    ]
    
    async def analyze_sentiment(text: str, provider_name: str) -> Dict[str, Any]:
        """Analyze sentiment of a text using a specific provider."""
        try:
            factory = manager.get_provider(provider_name)
            
            messages = [
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this text and respond with 'positive', 'negative', or 'neutral': {text}"
                }
            ]
            
            response = await factory.get_response_content_async(messages)
            return {
                "text": text,
                "sentiment": response.strip().lower(),
                "provider": provider_name,
                "success": True
            }
        except Exception as e:
            return {
                "text": text,
                "error": str(e),
                "provider": provider_name,
                "success": False
            }
    
    # Process all texts concurrently
    tasks = []
    for i, text in enumerate(texts):
        # Distribute across different providers
        provider_names = list(manager.providers.keys())
        if provider_names:
            provider = provider_names[i % len(provider_names)]
            tasks.append(analyze_sentiment(text, provider))
    
    if tasks:
        results = await asyncio.gather(*tasks)
        
        print("✅ Batch processing results:")
        for result in results:
            if result['success']:
                print(f"  '{result['text']}' -> {result['sentiment']} ({result['provider']})")
            else:
                print(f"  ERROR: '{result['text']}' -> {result['error']} ({result['provider']})")
    else:
        print("❌ No providers available for batch processing")


def example_provider_comparison():
    """Example: Compare responses from different providers."""
    print("\n=== Provider Comparison Example ===")
    
    manager = LLMFactoryManager()
    
    question = "What are the main benefits of renewable energy?"
    
    messages = [
        {
            "role": "user",
            "content": question
        }
    ]
    
    print(f"Question: {question}")
    print("\nResponses from different providers:")
    
    for provider_name, factory in manager.get_all_providers().items():
        try:
            response = factory.get_response_content(messages)
            print(f"\n--- {provider_name.upper()} ---")
            print(response[:200] + "..." if len(response) > 200 else response)
        except Exception as e:
            print(f"\n--- {provider_name.upper()} ---")
            print(f"❌ Failed: {e}")


def example_fallback_strategy():
    """Example: Fallback strategy when providers fail."""
    print("\n=== Fallback Strategy Example ===")
    
    manager = LLMFactoryManager()
    
    # Try providers in order of preference
    preferred_order = ['openai', 'anthropic', 'gemini', 'groq']
    
    messages = [
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ]
    
    for provider_name in preferred_order:
        try:
            factory = manager.get_provider(provider_name)
            response = factory.get_response_content(messages)
            print(f"✅ Successfully used {provider_name} provider:")
            print(response[:200] + "..." if len(response) > 200 else response)
            break
        except Exception as e:
            print(f"❌ {provider_name} provider failed: {e}")
            continue
    else:
        print("❌ All providers failed")


def main():
    """Run all examples."""
    print("=== General LLM Factory Practical Examples ===")
    print("Demonstrating real-world usage patterns with multiple providers.\n")
    
    # Run synchronous examples
    example_document_analysis()
    example_person_extraction()
    example_product_review_analysis()
    example_code_review()
    example_provider_comparison()
    example_fallback_strategy()
    
    # Run asynchronous examples
    print("\n=== Running Async Examples ===")
    asyncio.run(example_async_batch_processing())
    
    print("\n=== Examples Complete ===")
    print("These examples demonstrate the flexibility and power of the General LLM Factory.")


if __name__ == "__main__":
    main() 