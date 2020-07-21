# Source Anaconda
source /home/chuang/anaconda3/bin/activate nlp

# Move into right folder
cd src_ft

python config.py
#python 07_02_frequency_eval_aggregate.py -gsf grouped_search_words_final.csv -tn 15
python 07_02_frequency_eval_aggregate.py -gsf corporate_words_grouped.csv -tn 15

python compare_classification_to_csv.py