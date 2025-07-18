

#%%
import json,os,sys,time,warnings,glob   
import asyncio
from asyncio.sslproto import SSLAgainErrors
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import nest_asyncio
# Suppress warnings
warnings.filterwarnings("ignore")
nest_asyncio.apply()

# Add project directories to PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
LIBS_DIR = PROJECT_ROOT / "libs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
for subdir in (LIBS_DIR, PROMPTS_DIR):
    if str(subdir) not in sys.path:
        sys.path.insert(0, str(subdir))

# Local imports
from llm_factory_openai import BatchAsyncLLMAgent
from prompt_utils import load_prompt, format_messages  # type: ignore (local lib import)
from schemas import CountryIdentificationResponse 
from pydantic import BaseModel
from utils import read_json  # type: ignore (local lib import)

async def _build_batch_messages_from_articles(articles: List[Dict[str, Any]], 
                                              prompt_template:Dict,
                                              max_article_length=2000) -> List[List[Dict[str, str]]]:
    
    batch_messages: list[list[dict[str, str]]] = []
    batch_messages_ids: list[str] = []
    for art in articles:
        text_parts = [str(art.get(k, "")) for k in ("title", "snippet", "body") if art.get(k)]
        article_text = " ".join(text_parts)
        article_id = str(art.get("an", "unknown_id"))
        temp_promt_template = prompt_template.copy()
        temp_promt_template["user"] = temp_promt_template["user"].format(ARTICLE_CONTENT=article_text[:max_article_length])  # keep safety truncate
        messages = format_messages(temp_promt_template, add_schema=True)
        batch_messages.append(messages)
        batch_messages_ids.append(article_id)

    return batch_messages,batch_messages_ids
#%%
async def process_articles_async(batch_agent: BatchAsyncLLMAgent,
                                  batch_messages: List[List[Dict[str, str]]], 
                                  batch_size: int = 16,
                                  max_tokens: int = 2000,
                                  results_post_process = lambda x: x,
                                  **kwargs) -> List[str]:
    
    country_lists = await batch_agent.get_batch_response_contents_auto(batch_messages, batch_size=batch_size,
                                                                       max_tokens=max_tokens,**kwargs)
    country_lists = results_post_process(country_lists)

    return country_lists


def merge_ids_with_responses(
    ids: List[str],
    responses: List[Any],
) -> List[Dict[str, Any]]:
    """Merge article ids with LLM responses.

    Ensures robustness when a response is not a valid
    pydantic `BaseModel` instance (e.g., the raw string
    "LLM_ERROR").  Returns a list of dictionaries that are safe to
    serialise to JSON.
    """

    merged: list[dict[str, Any]] = []
    for msg_id, content in zip(ids, responses):
        # Accept any pydantic model (flexible)
        if isinstance(content, BaseModel):
            record: dict[str, Any] = content.dict()
        else:
            record = {
                "main_country": "",
                "other_countries": [],
                "error": str(content),
            }
        record["id"] = msg_id
        merged.append(record)

    return merged
#%%

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF fork warning

    parser = argparse.ArgumentParser(description="LLM country extraction")
    parser.add_argument("--data_dir", type=str, default="/ephemeral/home/xiong/data/Fund/Factiva_News/2025")
    parser.add_argument("--output_dir", type=str, default="/ephemeral/home/xiong/data/Fund/Factiva_News/results")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--test", type=bool, default=False)  # if True, only process the first 20 articles
    args = parser.parse_args([])
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    #%%
    local_model_args = {"model":"Qwen/Qwen3-8B",
                    "base_url":"http://localhost:8102/v1",
                    "temperature":0,
                    "api_key":"abc"
                    }
    batch_agent = BatchAsyncLLMAgent(**local_model_args)
    asyncio.run(batch_agent.test_connection())
    # Load the prompt template 
    prompt_path = "../prompts/extract_country_name.md"
    prompt_template = load_prompt(prompt_path).sections
    
    #%%
    start_time = time.time()

    json_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))  
    if args.test:
        json_files = json_files[:1]
        
    # Use tqdm to track progress over json_files
    for json_file in tqdm(json_files, desc="Processing JSON files", unit="file"):
        print(f"Processing file: {json_file}")
        articles = read_json(json_file)
        if args.test:
            articles = articles[:20]
        if not articles:
            print(f"  No articles found in {json_file}, skipping.")
            continue
        batch_messages,batch_messages_ids = asyncio.run(_build_batch_messages_from_articles(articles, prompt_template))
        country_lists = asyncio.run(process_articles_async(batch_agent, batch_messages, 
                                                           batch_size=1280,max_tokens=4000,
                                                           response_format=CountryIdentificationResponse,
                                                           safe_mode=True))
        assert len(batch_messages_ids) == len(country_lists)
        # Merge ids with responses using the helper for cleaner logic
        results_with_ids = merge_ids_with_responses(batch_messages_ids, country_lists)
        ## save results to json file
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(json_file))[0]
        # Create output filename
        output_file = os.path.join(args.output_dir, f"{base_filename}_countries_llm.json")
        # Save results_with_ids as a JSON file that can be loaded by read_json (list of dicts)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_with_ids, f, ensure_ascii=False, indent=2)
        print(f"Results with IDs saved to {output_file}")
        
    print("file processed, time taken: ", time.time() - start_time)
    
    #%%
    # ## debugging with a test file
    # test_file_path = args.data_dir + "/2025_articles_1.json"
    # articles = read_json(test_file_path)
    # print("Building batch messages")
    # batch_messages,batch_messages_ids = asyncio.run(_build_batch_messages_from_articles(articles, prompt_template))
    # print(batch_messages_ids[:10])
    # #%%
    # # # print(batch_messages[0][0]['content'])
    # # # print(batch_messages[0][1]['content'])
    # # #res = await batch_agent.get_batch_response_contents_auto(batch_messages[:20], batch_size=10,max_tokens=500)
    # res = asyncio.run( process_articles_async(batch_agent, batch_messages[:20], batch_size=20,max_tokens=2000,
    #                                         response_format=CountryIdentificationResponse, safe_mode=True))
    
    # # INSERT_YOUR_CODE
    # results_with_ids = [
    #     {**content.dict(), "id": msg_id}
    #     for msg_id, content in zip(batch_messages_ids[:20], res)
    # ]
    
    # # INSERT_YOUR_CODE
    # # Save results_with_ids as a JSON file that can be loaded by read_json (list of dicts)
    # output_json_path = os.path.join(args.output_dir, "2025_articles_1_countries_llm.json")
    # with open(output_json_path, "w", encoding="utf-8") as f:
    #     json.dump(results_with_ids, f, ensure_ascii=False, indent=2)
    # print(f"Results with IDs saved to {output_json_path}")


    # print(res)
    # #%%
    # country_lists = asyncio.run(process_articles_async(batch_agent, batch_messages,batch_size=1280,safe_mode=True))
    # print("Processing completed")
    # print(country_lists)
    # print("file processed, time taken: ", time.time() - start_time)
    # print(batch_messages)

# %%
