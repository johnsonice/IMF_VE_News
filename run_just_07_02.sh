# Source Anaconda
source /home/chuang/anaconda3/bin/activate nlp

# Move into right folder
cd src_ft

# Save current config file
cp config.py config_temp.py

# Run 07_02 with the specific config needed
rm config.py
cp config_topic_words.py config.py
python config.py
python 07_02_frequency_eval_aggregate.py -gsf final_topic_words_small.csv -tn 10

rm config.py
cp config_expert_words.py config.py
python config.py
python 07_02_frequency_eval_aggregate.py -gsf experts_refined.csv -tn 15

# Revert to original config
rm config.py
cp config_temp.py config.py