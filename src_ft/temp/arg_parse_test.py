"""
frequency_eval.py

Description: Used to evaluate supplied terms and term groups wrt recall, precision, and f2
based on whether or not the quarterly term freq is spiking significantly during the lead
up to crisis.

usage: python3 frequency_eval.py <TERM1> <TERM2> ...
NOTE: to see an explanation of optional arguments, use python3 frequency_eval.py --help
"""
import sys
sys.path.insert(0,'../libs')
sys.path.insert(0,'..')
import argparse
from crisis_points import crisis_points
import os
import config

#%%

if __name__ == '__main__':
    
    ## load config arguments
#    args = args_class(targets=config.targets,frequency_path=config.FREQUENCY,
#                          countries = config.countries,wv_path = config.W2V,
#                          sims=config.SIM,period=config.COUNTRY_FREQ_PERIOD, 
#                          months_prior=config.months_prior,
#                          window=config.smooth_window_size,
#                          eval_end_date=config.eval_end_date,
#                          weighted= config.WEIGHTED,
#                          z_thresh=config.z_thresh)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--targets', action='store', dest='targets', default=config.targets)
    parser.add_argument('-f', '--frequency_path', action='store', dest='frequency_path', default=config.FREQUENCY)
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=config.countries)
    parser.add_argument('-wv', '--wv_path', action='store', dest='wv_path', default=config.W2V)
    parser.add_argument('-ep', '--eval_path', action='store', dest='eval_path', default=config.EVAL_WG)
    parser.add_argument('-md', '--method', action='store', dest='method', default='zscore')
    parser.add_argument('-cd', '--crisis_defs', action='store', dest='crisis_defs', default='kr')
    parser.add_argument('-sims', '--sims', action='store_true', dest='sims', default=config.SIM)
    parser.add_argument('-tn', '--topn', action='store', dest='topn', default=config.topn)    
    parser.add_argument('-p', '--period', action='store', dest='period', default=config.COUNTRY_FREQ_PERIOD)
    parser.add_argument('-mp', '--months_prior', action='store', dest='months_prior', default=config.months_prior)
    parser.add_argument('-w', '--window', action='store', dest='window',default=config.smooth_window_size)
    parser.add_argument('-eed', '--eval_end_date', action='store', dest='eval_end_date',default=config.eval_end_date)
    parser.add_argument('-wed', '--weighted', action='store_true', dest='weighted',default=config.WEIGHTED)
    parser.add_argument('-z', '--z_thresh', action='store_true', dest='z_thresh',default=config.z_thresh)
    parser.add_argument('-gsf', '--search_file', action='store', dest='search_file',default=config.GROUPED_SEARCH_FILE)
    args = parser.parse_args()

    if args.sims:
        print('use sim')
        
    print(args.search_file)
    

