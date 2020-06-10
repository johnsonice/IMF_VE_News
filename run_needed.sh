source /home/chuang/anaconda3/bin/activate nlp
cd src_ft
pyhton config.py

python 02_1_meta_summary.py
python 06_0_frequency_country_specific_freqs.py
python 06_1_export_keywords_timeseries.py

python 07_02_frequency_eval_aggregate.py -gsf final_topic_words_small.csv -tn 10
python 07_02_frequency_eval_aggregate.py -gsf experts_refined.csv -tn 15