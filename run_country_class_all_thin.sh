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