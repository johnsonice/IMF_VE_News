source /home/chuang/anaconda3/bin/activate nlp
cd src_ft
python config.py

python 06_0_frequency_country_specific_freqs.py
python 06_1_export_keywords_timeseries.py

python 07_02_frequency_eval_aggregate.py -gsf grouped_search_words_final.csv -tn 15
python compare_class_types.py