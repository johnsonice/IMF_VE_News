cd src_ft
# Make sure all requisite folders exist
python config.py

# Evaluate the indices
python 07_02_frequency_eval_aggregate.py
# Compare the assessments
python 07_05_compare_classification_to_csv.py