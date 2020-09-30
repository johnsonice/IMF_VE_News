import sys,os
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import config
import crisis_points
import pandas as pd

combined_freqs_folder = '/data/News_data_raw/FT_WD_research/frequency/temp/All_Comb'
done_countries = set()
files_to_read = list(os.walk(combined_freqs_folder))[0][2]
for file_name in files_to_read:
    country = file_name.split('_')[0]
    done_countries.add(country)

imf_countries = set(crisis_points.imf_all_events.keys())

missing_countries = list(imf_countries - done_countries)
print('The Missing IMF Countries are: \n\n\n{}'.format(missing_countries))






