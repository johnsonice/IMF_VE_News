import asyncio
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
from llm_factory_openai import AsyncLLMAgent, BatchAsyncLLMAgent

#%%

async def call_llm(prompt, client):
    resp = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

async def main(prompts: list[str]):
    client = AsyncLLMAgent().client
    tasks = [call_llm(p, client) for p in prompts]
    # asyncio.gather returns results in the order of tasks → input order preserved
    results = await asyncio.gather(*tasks)
    return results


if __name__ == "__main__":
    
    prompts = ["Explain quantum computing in 200 words", "Write a sentence in 50 words", "Hi, how are you?"]
    
    # print("Using AsyncLLMAgent")
    # results = asyncio.run(main(prompts))
    # for prompt, answer in zip(prompts, results):
    #     print("-"*100)
    #     print(prompt, "→", answer)
    
    # print("-"*100)
    # use BatchAsyncLLMAgent
    print("Using BatchAsyncLLMAgent")
    batch_agent = BatchAsyncLLMAgent()
    batch_messages = [[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": p}] for p in prompts]
    results = asyncio.run(batch_agent.get_batch_response_contents(batch_messages))
    for prompt, answer in zip(prompts, results):
        print("-"*100)
        print(prompt, "→", answer)
        
    print("-"*100)
    print("Using BatchAsyncLLMAgent with auto batching")
    # Generate some random short prompts and specify a batch size for testing
    short_prompts = [
        "What's the capital of Italy?",
        "Define AI.",
        "Tell me a joke.",
        "Summarize the news.",
        "Translate 'hello' to French.",
        "List three colors.",
        "What is 2+2?",
        "Who wrote Hamlet?",
        "Give a synonym for 'happy'.",
        "What is the tallest mountain?"
    ]*10
    # Randomly select 5 prompts for this test
    batch_size = 8  # Example batch size for auto batching
    batch_messages = [[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": p}] for p in short_prompts]
    results = asyncio.run(batch_agent.get_batch_response_contents_auto(batch_messages, batch_size=batch_size))
    # for prompt, answer in zip(short_prompts, results):
    #     print("-"*100)
    #     print(prompt, "→", answer)
    