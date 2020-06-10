source /home/chuang/anaconda3/bin/activate nlp
cd src_ft

python 07_02_frequency_eval_aggregate.py -gsf final_topic_words_small.csv -tn 10 > ~/logs/log_07_02_1_tt.txt
python 07_02_frequency_eval_aggregate.py -gsf experts_refined.csv -tn 15 > ~/logs/log_07_02_1_tt.txt