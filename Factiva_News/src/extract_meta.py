#%%
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
import sys
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
# Add parent directory to path for imports|
sys.path.append(str(Path(__file__).parent.parent))
from libs.utils import read_json
from libs.meta_utils import construct_country_group_rex,tag_country
from libs.country_dict_full import get_dict
#%%
class Article_Transformer:
    """
    Class to transform article metadata.
    """
    def __init__(self, country_rex_dict):
        self.country_rex_dict = country_rex_dict

    @staticmethod
    def transform_dates(article):
        """Transform date/datetime fields to readable format"""
        date_patterns = {'date', 'datetime', 'time', 'timestamp', 'created', 'modified', 
                        'published', 'ingestion', 'publication', 'modification'}
        
        for key, value in article.items():
            if (any(p in key.lower() for p in date_patterns) and 
                isinstance(value, str) and value.isdigit()):
                try:
                    ts = int(value) / (1000 if len(value) == 13 else 1)
                    article[key] = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, OSError):
                    continue
        return article

    def identify_countries(self,article):
        """
        Identify countries mentioned in the article using regex patterns.
        Parameters:
            article (dict): Article dictionary containing text fields.
            country_rex_dict (dict): Dictionary of regex patterns for country names.
        Returns:
            list: List of identified countries.
        """
        if not isinstance(article, dict):
            return []
        
        country_tags = list(tag_country(article,self.country_rex_dict))
        article['country_tags_rulebased'] = country_tags
        
        return article
    
    def __call__(self, article):
        # Transform date fields
        article = self.transform_dates(article)
        # Identify countries
        article = self.identify_countries(article)
        
        return article

def extract_metadata(articles,
                     original_filename="", 
                     other_transform_func=None,
                     verbose=0,
                     n_jobs=-1
                     ):
    """
    Extract metadata from articles into a DataFrame with parallel processing support.
    
    Parameters:
        articles (list): List of article dictionaries.
        original_filename (str, optional): Source filename. Defaults to "".
        other_transform_func (callable, optional): Additional transformation function. Defaults to None.
        verbose (bool, optional): Print processing information. Defaults to True.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to 1.
                               -1 means using all processors.
    
    Returns:
        pd.DataFrame: DataFrame containing article metadata (text content removed).
    """
    def process_single_article(article):
        if not isinstance(article, dict):
            return None
        # Create copy and remove text fields
        meta = article.copy()
        # process all transformations when provided 
        if other_transform_func:
            if callable(other_transform_func):
                meta = other_transform_func(meta)
            
        meta['original_filename'] = original_filename
        meta.pop('body', None)
        meta.pop('snippet', None)
        return meta
    
    # Process articles in parallel
    processed = Parallel(n_jobs=n_jobs,verbose=verbose)(
        delayed(process_single_article)(article) for article in articles
    )
    
    # Filter out None values
    processed = [meta for meta in processed if meta is not None]
    
    df = pd.DataFrame(processed)
    if verbose:
        print(f"Processed {len(articles)} articles into DataFrame")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
    return df

def process_directory(data_dir, 
                      country_rex_dict,
                      article_transformer,
                      output_file=None, 
                      n_jobs=-1,
                      sub_n_jobs=8,
                      verbose=1):
    """Process all JSON files in directory and extract metadata using parallel processing"""
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json"))
    
    def process_single_file(file_path):
        """Process a single JSON file and return metadata DataFrame"""
        print(f"Processing {file_path.name}...")
        articles = read_json(file_path)
        
        if articles:
            metadata_df = extract_metadata(articles, 
                                           original_filename=file_path.name,
                                           other_transform_func=article_transformer,
                                           n_jobs=sub_n_jobs)
            print(f"  Extracted {len(metadata_df)} articles from {file_path.name}")
            return metadata_df
        return pd.DataFrame()
    
    # Process files in parallel
    all_metadata = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_single_file)(file_path) for file_path in json_files
    )
    
    # Filter out empty DataFrames and combine
    all_metadata = [df for df in all_metadata if not df.empty]
    
    if all_metadata:
        combined_df = pd.concat(all_metadata, ignore_index=True)
        print(f"\nTotal articles: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        
        if output_file:
            combined_df.to_csv(output_file, index=False)
            print(f"Saved metadata to {output_file}")
        
        return combined_df
    
    return pd.DataFrame()

def unit_test_transformer():
    """Unit test for Article_Transformer"""
    country_dict = get_dict()
    country_rex_dict = construct_country_group_rex(country_dict)
    transformer = Article_Transformer(country_rex_dict)
    
    # Sample article
    article = {
        "title": "US Economy Grows",
        "body": "The US economy is growing rapidly.",
        "date": "20250101",
        "country_tags_rulebased": []
    }
    
    transformed_article = transformer(article)
    print(transformed_article)
    
    assert transformed_article['country_tags_rulebased'] == ['united states of america']
    
    print("Unit test passed!")

def unit_test_extract_metadata():
    
    country_dict = get_dict()
    country_rex_dict = construct_country_group_rex(country_dict)
    article_transformer = Article_Transformer(country_rex_dict)
    
    test_dir = data_dir+"/2025_articles_1.json"
    articles = read_json(test_dir)
    if articles:
        metadata_df = extract_metadata(articles, original_filename=test_dir, 
                                       other_transform_func=article_transformer)
        print(f"\nExtracted metadata from {len(metadata_df)} articles in {test_dir}")
        print(metadata_df.head())
    else:
        print(f"No articles found in {test_dir}")
        
    return metadata_df

#%%
if __name__ == "__main__":
    # Process Factiva News data
    data_dir = "/ephemeral/home/xiong/data/Fund/Factiva_News/2025"
    output_file = "/ephemeral/home/xiong/data/Fund/Factiva_News/2025_metadata.csv"
    
    country_dict = get_dict()
    country_rex_dict = construct_country_group_rex(country_dict)
    article_transformer = Article_Transformer(country_rex_dict)
        
    metadata_df = process_directory(data_dir,country_rex_dict,article_transformer, output_file=output_file, n_jobs=4,sub_n_jobs=16) # output_file
    
    if not metadata_df.empty:
        print(f"\nDataFrame shape: {metadata_df.shape}")
        print("\nSample metadata:")
        print(metadata_df.head())


# %%
