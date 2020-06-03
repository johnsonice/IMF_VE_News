cd src_ft

#Setup test ENV
cp config.py config_temp.py
rm config.py
cp config_tiny_test.py config.py

#Run the test
python 01_corpus_preprocessing.py
python 02_0_corpus_doc_details.py
python 02_1_meta_summary.py
#python 03_corpus_phrases.py
#python 04_1_vectorize_words.py
#python 06_0_frequency_country_specific_freqs.py
#python 06_1_export_keywords_timeseries.py
#python 07_02_frequency_eval_aggregate.py -gsf final_topic_words_small.csv -tn 10
#python 07_02_frequency_eval_aggregate.py -gsf experts_refined.csv -tn 15

#Return standard ENV
rm config.py
cp config_temp.py config.py


