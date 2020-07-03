import os
import sys
import pandas as pd
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config
from glob import glob
from shutil import copy2

model_name = "ldaviz_t100"
temp_pkl_file = "/home/apsurek/IMF_VE_News/src_ft/temp_in_processing.pkl"
ds = pd.read_pickle(temp_pkl_file)
# os.remove("temp_in_processing.pkl") # put into final

meta_root = config.DOC_META
meta_aug = os.path.join(config.AUG_DOC_META, 'doc_details_crisis_aug_{}.pkl'.format('Min1'))
meta_pkl = config.DOC_META_FILE

df = pd.read_pickle(meta_pkl)  # Re-load deleted df - not multiplied when multiprocessing anymore
new_df = df.join(ds)  # merge country meta
new_df_file = os.path.join(meta_aug, 'doc_details_{}_topic_{}.pkl'.format('crisis', model_name))
new_df.to_pickle(new_df_file)
print('Topic document meta data saved at {}'.format(new_df_file))

aug_df = pd.read_pickle(meta_aug)
new_aug_df = aug_df.join(ds)
new_aug_file = os.path.join(meta_aug, 'doc_details_{}_aug_{}_topic_{}.pkl'.format('crisis', 'Min1', model_name))
new_aug_df.to_pickle(new_aug_file)
print('Aug topic document meta data saved at {}'.format(new_aug_file))
