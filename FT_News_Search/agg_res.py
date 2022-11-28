## evaluate results 

#%%
import os,sys
import pandas as pd
sys.path.insert(0,'libs')
from utils import load_pickle,get_all_files
import numpy as np
#%%
def process_raw_search_res(input_list):
    df = pd.DataFrame(input_list)
    meta_columns = df.columns[:7]
    match_columns = df.columns[7:]
    df[match_columns] = np.where(df[match_columns]>0,1,0) ## convert everything to 0 and 1
    agg_df = df.groupby(['month','quarter','year'])[match_columns].sum()
    agg_df.reset_index(inplace=True)

    return agg_df

def process_one_year(pickle_file):
    sr = load_pickle(pickle_file)
    agg_df_y = process_raw_search_res(sr)
    return agg_df_y


#%%
if __name__=="__main__":
    wd_path = '/data/chuang/Financial_Times/Working_directory'
    res_path = os.path.join(wd_path,'search_results')
    #res_search = load_pickle(os.path.join(wd_path,'search_results','search_raw.pkl'))
    #%%
    res_files = get_all_files(res_path)
    res_files = [r for r in res_files if 'search_raw.pkl' not in r]

    all_agg_dfs = []
    for res_f in res_files:
        print(res_f)
        a_df=process_one_year(res_f)
        all_agg_dfs.append(a_df)
    #%%
    merged_df = pd.concat(all_agg_dfs,ignore_index=True)
    merged_df.fillna(0,inplace=True)

    ## reorder columns and sort 
    cols = merged_df.columns.tolist()
    sorted_col = cols[3:]
    sorted_col.sort()
    cols = cols[:3]+ sorted_col
    merged_df = merged_df[cols]
    merged_df.sort_values(by=['month','quarter','year'],inplace=True)
    ## export to csv
    merged_df.to_csv((os.path.join(wd_path,'ft_company_match_agg.csv')))
    print('export aggregated daata to {}'.format(os.path.join(wd_path,'ft_company_match_agg.csv')))
# %%
    # USE_MERGED=False
    # #%%
    # if USE_MERGED:
    #     ## save raw match results into files 
    #     res_df = pd.DataFrame(res_search)
    #     #res_df.to_csv(os.path.join(wd_path,'ft_company_match.csv'))
    #     #res_df.to_pickle(os.path.join(wd_path,'ft_company_match.pkl'))
    #     #print('export to {}'.format(wd_path))
    #     ## process raw results 
    #     #print(res_df.columns)
    #     meta_columns = res_df.columns[:7]
    #     match_columns = res_df.columns[7:]
    #     res_df.fillna(0,inplace=True)
    #     #res_df[match_columns] = np.where(res_df[match_columns]>0,1,0) ## this can take really long
    #     for mc in match_columns: ## try this instead
    #         print(mc)
    #         res_df[mc] = np.where(res_df[mc]>0,1,0)
    #     print(meta_columns)
    #     agg_df = res_df.groupby(['month','quarter','year'])[match_columns].sum()
    #     agg_df.to_csv(os.path.join(wd_path,'ft_company_match_agg.csv'))
    # else:
