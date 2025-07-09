"""
Unit tests for SimpleLLMAgent structured output functionality.

Setup Instructions:
1. Create a .env file in the IMF_VE_News directory (one level up from Factiva_News)
2. Add your OpenAI API key: OPENAI_API_KEY=your_key_here
3. Get your API key from: https://platform.openai.com/api-keys

"""
#%%
import sys
import os
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

# Import SimpleLLMAgent from libs directory
from llm_factory_openai import SimpleLLMAgent

def unit_test_structured_output():
    """Test structured output with Pydantic models."""
    try:
        agent = SimpleLLMAgent()
        print("✅ SimpleLLMAgent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize SimpleLLMAgent: {e}")
        print("💡 Make sure OPENAI_API_KEY environment variable is set")
        print("💡 Or create a .env file with OPENAI_API_KEY=your_key_here")
        return
    
    # Example using Pydantic for structured output
    try:
        # Example using Pydantic for structured output
        class Capital(BaseModel):
            city: str = Field(..., description="Name of the capital city")
            country: str = Field(..., description="Name of the country")
            population: int = Field(..., description="Population count")
            landmarks: List[str] = Field(default_factory=list, description="Notable landmarks")

        class CapitalsResponse(BaseModel):
            capitals: List[Capital] = Field(default_factory=list)
        
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
        
        # Test 2: Direct Pydantic model via response_format
        print("1. Using response_format with Pydantic model:")
        capitals_direct = agent.get_response_content(messages, response_format=CapitalsResponse)
        print("✅ Direct Pydantic response successful:")
        for capital in capitals_direct.capitals:
            print(f"  {capital.city}, {capital.country}")
            print(f"  Population: {capital.population:,}")
            print(f"  Landmarks: {', '.join(capital.landmarks)}")
            print()
        
        # Test 3: JSON mode with manual parsing (backward compatibility)
        print("2. Using JSON mode with manual parsing:")
        messages_json = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides structured data about European capitals."
            },
            {
                "role": "user",
                "content": """List 2 European capitals with their details. 
                Return as JSON matching this structure:
                {
                    "capitals": [
                        {
                            "city": "string",
                            "country": "string", 
                            "population": 123456,
                            "landmarks": ["array of strings"]
                        }
                    ]
                }"""
            }
        ]
        
        response_json = agent.get_response_content(messages_json, response_format={"type": "json_object"})
        parsed_data = agent.parse_json(response_json)
        capitals_data = CapitalsResponse.model_validate(parsed_data)
        
        print("✅ JSON parsing response successful:")
        for capital in capitals_data.capitals:
            print(f"  {capital.city}, {capital.country}")
            print(f"  Population: {capital.population:,}")
            print(f"  Landmarks: {', '.join(capital.landmarks)}")
            print()
            
    except Exception as e:
        print(f"❌ Structured output failed: {e}")

#%%
if __name__ == "__main__":
    print("=== Running Unit Tests ===")
    unit_test_structured_output()
    print("=== Unit Tests Complete ===")
# %%
