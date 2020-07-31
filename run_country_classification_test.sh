# Load anaconda environment
source /home/chuang/anaconda3/bin/activate nlp
cd src_ft
# Make sure all requisite folders exist
python config.py

# Classify documents by country
python 02_1_meta_summary.py

# Count word frequencies
python 06_0_frequency_country_specific_freqs.py
# Generate frequency-in-time-period time series
python 06_1_export_keywords_timeseries.py

# Evaluate predictive power of time series based on the passed word grouping
python 07_02_frequency_eval_aggregate.py -gsf grouped_search_words_final.csv -tn 15

# Create a comparison table to assess best county identification scheme
python compare_classification_to_csv.py