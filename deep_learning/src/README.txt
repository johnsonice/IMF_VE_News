### read me 

- config : all arguments; folder pathes etc

1) feature_extractor.py : use bert base to encode all article titles and snips of countries we cover 

2) process_crisis_dates.py [_*]: -transform raw crisis data from excel to pandas df pickle object 
                                 -we also transformed the data to crisis/precrisis/trainquile periods 
                                 -specific precrisis winidow size is defiend in the script

3) training_data_prepare.py : - get all news embedings and merged with crisis date, so that we have different versions of training data ready
                              - there are argumetns in the script to toggle which crisis dates versions to run  
                              
4) model_training_simple : - simple random split training  
   model_training_cv : - 5 fold cross validation training using country groups 

5) model_inference_simple: inference and test on all data we have 
   model_inference_cv: inference and test using a cross validation fashion

   

