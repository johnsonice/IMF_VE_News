import os
from pandas import DataFrame
import datetime
import math
import glob

if __name__ == "__main__":

    ## Save the event that we are interested in text for now ->> datetime object later
    event_date = "2018-05-08" # Starting with Argentina
    event_name = "Argentina, May 2018"
    
    # Temporary
    split_date = event_date.split("-")

    year_of_event = int(split_date[0])
    month_of_event = int(split_date[1])
    day_of_event = int(split_date[2])

    # Mutable window for discovery
    months_prior = 18
    
    # includes partial first month and the entire month of event if True
    partial_months_as_whole = True

    # Months to read data from
    if partial_months_as_whole:
        months_prior += 1
    start_year = year_of_event - math.ceil (months_prior - month_of_event)/12)
    start_month = 12 - (months_prior-month_of_event)%12 + 1
    start_day = day_of_event

    # Generate list of folders to look in - only two top-level raw data folders on the instance, glob the subdirectories
    top_level_folders = ["/data/News_data_raw/Financial_Times/all_current/", "/data/News_data_raw/Financial_Times/all_18m6_19m4/"]
    all_folders = []
    for fold in top_level_folders:
        all_folders = all_folders + glob.glob(fold)
    
    # Only keep folders for the proper years
    folders_to_read = []
    years_of_interest = range(start_year, year_of_event+1)
    for year in years_of_interest:
        for folder_name in all_folders:
            if str(year) in folder_name:
                folders_to_read.append(folder_name)
    
    # Generate list of files to read, with year, month, in dataframe -> will use to create a csv with country-classification tally
    files_to_read  = []
    years_temp = []
    months_temp = []
    ended = False
    current_year = start_year
    current_month = start_month
    
    # Go through the folders and pick out names of files to read - direct referrences
    while  not ended:

        # From naming convention
        file_name_variety = "*"+current_year+"-"+current_month

        # Generate the file names
        files_with_name_variety = []
        for folder_name in folders_to_read:
            if str(year) in folder_name:
                files_with_name_variety = files_with_name_variety + glob.glob(folder_name+'\\'+file_name_variety+'*.json')

        files_to_read = files_to_read + files_with_name_variety
        years_temp = years_temp + [current_year]*len(files_with_name_variety)
        months_temp = months_temp + [current_month]*len(files_with_name_variety)

        # Control year, month
        current_month += 1
        if current_month%13 == 0:
            current_month = 1
            current_year+=1
        
        # End conditions (end of date range)
        if current_year == year_of_event:
            if not partial_months_as_whole and  current_month == month_of_event:
                ended = True
            elif current_month == month_of_event+1:
                ended = True
    
    # Write results to dataframe
    files_close_look = DataFrame({'year':years_temp, 'month':months_temp, 'file':files_to_read})

    

    


