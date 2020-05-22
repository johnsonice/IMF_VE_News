import os
import pandas as pd
# from pandas import DataFrame
import datetime
import math
import glob
import re
import random as rand
import numpy as np
from nltk import pos_tag
import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

random_samp = 100


def get_wordnet_pos(tag):
    # Return whether "tag" is adjective,
    # noun, verb or adverb
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

"""
def get_country_vocab(country,period='quarter',frequency_path=config.FREQUENCY):
    data_path = os.path.join(frequency_path,'{}_{}_word_freqs.pkl'.format(country, period))
    data = pd.read_pickle(data_path)
    vocab = list(data.index)
    return vocab
"""

if __name__ == "__main__":

    ## Save the event that we are interested in text for now ->> datetime object later
    event_date = "2018-05-08"  # Starting with Argentina
    event_name = "Argentina_May_2018"
    search_term = "Argentina"  # This is the search phrase used against article
    sample_size = 100  # Determine amount of articles picked to read by human or otherwise double-check - int absolute number
    sample_only_matches = True  # Sample only from articles containing the search phrase
    save_top_10 = True  # Saves the 10 articles that use the search phrase the most (separate from sampling)
    thick = True  # Enable saving og the article text - for verification by a human being

    country_list = pd.read_csv("country_list.csv", header=None)[0].values

    # Temporary
    split_date = event_date.split("-")

    event_log = open("test_log.txt", "w+")

    year_of_event = int(split_date[0])
    month_of_event = int(split_date[1])
    day_of_event = int(split_date[2])

    event_log.write(
        "Date split y/m/d: " + str(year_of_event) + "/" + str(month_of_event) + "/" + str(day_of_event) + "\n")

    # Mutable window for discovery
    months_prior = 18

    # includes partial first month and the entire month of event if True
    partial_months_as_whole = True

    # Months to read data from
    if partial_months_as_whole:
        months_prior += 1
    start_year = year_of_event - int(math.ceil((float(months_prior) - month_of_event) / 12))
    start_month = 12 - (months_prior - month_of_event) % 12 + 1
    start_day = day_of_event

    event_log.write("Start y/m/d: " + str(start_year) + "/" + str(start_month) + "/" + str(start_day) + "\n")

    # Generate list of folders to look in - only two top-level raw data folders on the instance, glob the subdirectories
    top_level_folders = ["/data/News_data_raw/Financial_Times/all_current/",
                         "/data/News_data_raw/Financial_Times/all_18m6_19m4/"]
    all_folders = []
    for fold in top_level_folders:
        all_folders = all_folders + glob.glob(fold + "*")

    event_log.write("Top Level Folders: " + str(top_level_folders) + "\n")
    event_log.write("All Folders: " + str(all_folders) + "\n")

    # Only keep folders for the proper years
    folders_to_read = []
    years_of_interest = range(start_year, year_of_event + 1)
    for year in years_of_interest:
        for folder_name in all_folders:
            if str(year) in folder_name:
                folders_to_read.append(folder_name)

    event_log.write("Kept folders: " + str(folders_to_read) + "\n")

    # Generate list of files to read, with year, month, in dataframe -> will use to create a csv with country-classification tally
    files_to_read = []
    years_temp = []
    months_temp = []
    counts_temp = []
    ended = False
    current_year = start_year
    current_month = start_month

    # Go through the folders and pick out names of files to read - direct references
    while not ended:

        # From naming convention
        if current_month < 10:
            file_name_variety = "*" + str(current_year) + "-0" + str(current_month) + "*.json"
        else:
            file_name_variety = "*" + str(current_year) + "-" + str(current_month) + "*.json"

        event_log.write("While, y/m: " + str(current_year) + "/" + str(current_month) + "\n")
        event_log.write("Variety:: " + file_name_variety + "\n\n")
        # Generate the file names
        files_with_name_variety = []
        for folder_name in folders_to_read:
            if str(current_year) in folder_name:
                files_with_name_variety = files_with_name_variety + glob.glob(folder_name + '/' + file_name_variety)

        files_to_read = files_to_read + files_with_name_variety
        years_temp = years_temp + [current_year] * len(files_with_name_variety)
        months_temp = months_temp + [current_month] * len(files_with_name_variety)

        # Control year, month
        current_month += 1
        if current_month % 13 == 0:
            current_month = 1
            current_year += 1

        # End conditions (end of date range)
        if current_year == year_of_event:
            if not partial_months_as_whole and current_month == month_of_event:
                ended = True
            elif current_month == month_of_event + 1:
                ended = True

    random_sample_indices = []

    random_sample_indices = rand.sample(range(len(files_to_read)), sample_size)

    # Write results to dataframe
    #   files_close_look = DataFrame({'year':years_temp, 'month':months_temp, 'file':files_to_read})

    top_10_indicies = []
    second_lowest_of_top_counts = 0
    lowest_of_top_counts = 0

    positive_file_indices = []

    # Stemmer selection
    stemmer = PorterStemmer()

    # Lemmatizer selection
    lemmatizer = WordNetLemmatizer()

    # Save list of files that need to be read - clears memory of the (potentially long) list
    files_read_csv = 'files_to_inspect_' + event_name + '.csv'

    event_stats_csv = 'stats_leading_up_to_' + event_name + '.csv'
    event_stats_file = open(event_stats_csv, 'w+')
    thick_event_stat_file = open("thick_"+event_stats_csv, 'w+')

    event_stat_header = 'year,month,absolute_name,search_country_count_first_50,search_country_count_first_100,' \
                        'search_country_count_total,other_country_count_first_50,other_country_count_first_100,' \
                        'other_country_count_total'
    event_stats_file.write(event_stat_header+'\n')

    thick_event_stat_file.write(event_stat_header + ",article_text\n")

    # Stem and Lemma the country_list for comparison
    country_list = [stemmer.stem(el) for el in country_list]

    # If Lemmaing
    country_list_pos = nltk.pos_tag(country_list)
    country_list = [lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) for el in country_list_pos]

    with open(files_read_csv, 'w+') as out_file:
        for ind in range(len(random_sample_indices)):
            i = random_sample_indices[ind]
            file_name = files_to_read[i]
            file_date = str(years_temp[i]) + ',' + str(months_temp[i])
            file_desc = file_date + ',' + file_name

            # Read the article
            with open(file_name, 'r') as article:
                article_contents = article.read()

            # Throw out html, xml, etc
            # HTML/XML Cleaning - throw out HTML, XML
            article_contents = re.sub(r"(<[^>]*>)", " ", article_contents)

            # Throw out digits, non-alphanum, lowercase
            article_contents = re.sub(r"([^\w]*)|(\d*)", " ", article_contents).lower()

            # Tokenize - numpy array (faster)
            #tokenized = np.char.split("\s").astype(str)

            # Dirty - swap to numpy later
            tokenized = article_contents.split()

            # Stem
            tokens_stemmed = [stemmer.stem(el) for el in tokenized]

            # Count Instances - first 50
            search_country_count_f50 = 0
            other_country_count_f50 = 0

            # Count in first 100
            search_country_count_f100 = 0
            other_country_count_f100 = 0

            # Count in document
            search_country_count = 0
            other_country_count = 0

            for j in range(len(tokens_stemmed)):
                inc_search = False
                inc_other = False
                tok = tokens_stemmed[j]

                if tok == search_term:
                    inc_search = True
                elif tok in country_list:
                    inc_other = True

                if j < 50:
                    search_country_count_f50 += int(inc_search)
                    other_country_count_f50 += int(inc_other)
                if j < 100:
                    search_country_count_f100 += int(inc_search)
                    other_country_count_f100 += int(inc_other)

                search_country_count += int(inc_search)
                other_country_count += int(inc_other)

            # Save file statistics
            file_stats = file_date + ',' + file_name + ',' \
                         + str(search_country_count_f50) + ',' + str(search_country_count_f100)\
                         + ',' + str(search_country_count) + ',' + str(other_country_count_f50) \
                         + ',' + str(other_country_count_f100) + ',' + str(other_country_count)

            # Did lots of processing! Want to save the results! -  if thick enabled
            if thick:
                glued = " ".join(tokens_stemmed)
                file_thick = file_stats + ',' + glued
                thick_event_stat_file.write(file_stats + "\n")

            event_stats_file.write(file_stats + "\n")

            if search_country_count > 0:
                positive_file_indices.append(i)

            out_file.write(file_desc + "\n")

    event_stats_file.close()

    if thick:
        thick_event_stat_file.close()

    event_log.write("File-list successfully written to: " + files_read_csv + "\n")

    # Third Order (elimination of irrelevant topics)
    # Only select positive for x topic
    # Throw away positives for x topic
