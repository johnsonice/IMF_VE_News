"""
DEFUNCT
"""

from frequency_utils import word_frequency, id2full_path, freq_increasing
import pandas as pd

term = "_t-1"
doc_details = pd.read_pickle('/home/ubuntu/Documents/v_e/data/doc_details_full{}.pkl'.format(term))
cris = doc_details[doc_details['crisis'] == 1].index
nocris = doc_details[doc_details['crisis'] == 0].index

cris_freq = word_frequency(id2full_path(cris), verbose=True)
cris_freq.to_pickle('/home/ubuntu/Documents/v_e/data/frequency/crisis_word_freqs{}.pkl'.format(term))

nocris_freq = word_frequency(id2full_path(nocris), verbose=True)
nocris_freq.to_pickle('/home/ubuntu/Documents/v_e/data/frequency/nocrisis_word_freqs{}.pkl'.format(term))

signif_increase = freq_increasing(nocris_freq, cris_freq)
signif_increase.to_pickle('/home/ubuntu/Documents/v_e/data/frequency/significant_crisis_freq_increase{}.pkl'.format(term))
