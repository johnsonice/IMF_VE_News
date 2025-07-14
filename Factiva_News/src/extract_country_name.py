

#%%
import asyncio
import time
import warnings
from pathlib import Path
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
# Add <project_root>/libs to PYTHONPATH so we can `import llm_factory_openai`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
LIBS_DIR = PROJECT_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from utils import read_json  # type: ignore (local lib import)
from llm_factory_openai import BatchAsyncLLMAgent
import nest_asyncio
nest_asyncio.apply()
#%%
# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
COUNTRY_PROMPT_TEMPLATE = {"system": "You are a news analyst and your job is to determind the main countries mentioned in the news article.\n", 
                           "user":
                            """
                            "Based on the content of the entire news article, please identify main countries being discussed in news article text.\n"
                            "Return ONLY a comma-separated list of country names in English (lower-case).\n"
                            "If no country is mentioned, return an empty string.\n\n"
                            "<article>\n{text}\n</article>
                            """}

async def _build_batch_messages_from_articles(articles: List[Dict[str, Any]], prompt_template: dict[str, str]) -> List[List[Dict[str, str]]]:
    
    batch_messages: list[list[dict[str, str]]] = []
    for art in articles:
        text_parts = [str(art.get(k, "")) for k in ("title", "snippet", "body") if art.get(k)]
        article_text = " ".join(text_parts)
        temp_promt_template = prompt_template.copy()
        temp_promt_template["user"] = temp_promt_template["user"].format(text=article_text[:4000])  # keep safety truncate
        batch_messages.append([{"role":"system","content":temp_promt_template["system"]},{"role":"user","content":temp_promt_template["user"]}])

    return batch_messages
#%%
async def process_articles_async(batch_agent: BatchAsyncLLMAgent,
                                  batch_messages: List[List[Dict[str, str]]], 
                                  batch_size: int = 16,
                                  results_post_process = lambda x: x) -> List[str]:
    
    country_lists = await batch_agent.get_batch_response_contents_auto(batch_messages, batch_size=batch_size,max_tokens=500)
    country_lists = results_post_process(country_lists)

    return country_lists
#%%

if __name__ == "__main__":
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence HF fork warning

    parser = argparse.ArgumentParser(description="LLM country extraction")
    parser.add_argument("--data_dir", type=str, default="/ephemeral/home/xiong/data/Fund/Factiva_News/2025")
    parser.add_argument("--output_file", type=str, default="/ephemeral/home/xiong/data/Fund/Factiva_News/2025_countries_llm.csv")
    parser.add_argument("--n_jobs", type=int, default=4)

    args = parser.parse_args([])
    
    local_model_args = {"model":"Qwen/Qwen3-8B",
                    "base_url":"http://localhost:8101/v1",
                    "temperature":0.2,
                    "api_key":"abc"
                    }
    batch_agent = BatchAsyncLLMAgent(**local_model_args)
    start_time = time.time()
    print("Processing test file 2025_articles_1.json")
    test_file_path = args.data_dir + "/2025_articles_1.json"
    articles = read_json(test_file_path)
    print("Building batch messages")
    batch_messages = asyncio.run(_build_batch_messages_from_articles(articles, COUNTRY_PROMPT_TEMPLATE))
    print("Processing articles")
    country_lists = asyncio.run(process_articles_async(batch_agent, batch_messages,batch_size=1280))
    print("Processing completed")
    print(country_lists)
    print("file processed, time taken: ", time.time() - start_time)
    #print(batch_messages)
