source /home/chuang/anaconda3/bin/activate nlp
cd src_ft
python config.py

python 10_01_topic_desc_aug_meta.py

python 06_0_frequency_country_specific_freqs.py
python 06_1_export_keywords_timeseries.py