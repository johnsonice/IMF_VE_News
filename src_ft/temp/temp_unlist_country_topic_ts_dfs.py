import sys,os
sys.path.insert(0,'..')
sys.path.insert(0,'../libs')
import config_topiccing as config
import pandas as pd
from stream import MetaStreamer_uberfast as MetaStreamer
from mp_utils import Mp
import pickle as pkl

if __name__ == '__main__':
    class_type_setups = config.class_type_setups
    countries = config.countries
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"

    for setup in class_type_setups:
        setup_name = setup[0]
        load_folder = os.path.join(topiccing_folder, 'time_series_bkp', setup_name)
        export_folder = os.path.join(topiccing_folder, 'time_series', setup_name)
        for country in countries:
            load_csv = os.path.join(load_folder, "{}_100_topic_time_series.csv".format(country))
            export_csv = os.path.join(export_folder, "{}_100_topic_time_series.csv".format(country))
            load_df = pd.read_csv(load_csv)
            for topic_num in range(100):
                load_df[str(topic_num)] = load_df[str(topic_num)].apply(lambda x: x[1:-1])
            load_df.to_csv(export_csv, index=False)
