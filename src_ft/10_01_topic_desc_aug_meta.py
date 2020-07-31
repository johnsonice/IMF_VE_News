import sys, os
sys.path.insert(0, './libs')
import config
import pandas as pd
from stream import MetaStreamer_fast as MetaStreamer

if __name__ == "__main__":
    setups = config.class_type_setups

    countries = config.countries
    topic_f2_thresholds = [('top', 1), ('top', 5),('top', 10), .5, .4, .3]
    #topic_f2_thresholds = [0]
    document_topic_min_levels = [("top", 1), ("top", 2), .5, .25, .1, .05, ("top", 10), ("top", 20), .02, .01]
    #document_topic_min_levels = [.00001]

    debug = False  # TEST

    '''
    if debug:
        countries = ['argentina']
        #topic_f2_thresholds = [('top', 1), .5]
        topic_f2_thresholds = [.5]
        document_topic_min_levels = [("top", 1)]
    '''

    num_topics = config.num_topics

    model_name = "ldaviz_t100"
    topiccing_folder = "/data/News_data_raw/FT_WD_research/topiccing"
    series_saved_at = os.path.join(topiccing_folder, '{}_topic_meta'.format(model_name))
    series_base_file = os.path.join(series_saved_at, "series_savepoint_part{}.pkl")

    partition_size = 200000
    num_of_series = len(list(os.walk(series_saved_at))[0][2])

    for topic_f2_thresh in topic_f2_thresholds:

        country_topic_dict = {}
        for country in countries:

            country_topic_info_file = os.path.join('/data/News_data_raw/FT_WD_research/topiccing/eval/Min1_AllCountry',
                                                   '{}_{}_topic_eval.csv'.format(country, num_topics))
            country_df = pd.read_csv(country_topic_info_file)

            if debug:
                print("COUNTRY DF HEAD")
                print(country_df.head())

            country_topics = []
            if type(topic_f2_thresh) is tuple:
                if topic_f2_thresh[0] == 'top':
                    top_n = topic_f2_thresh[1]

                    # Get first n of descneding value sort
                    country_df = country_df.sort_values(['fscore'], ascending=False)[:top_n]
                    country_topics = list(country_df.index.values)
            else:
                for i in range(num_topics):
                    if country_df.at[i, 'fscore'] >= topic_f2_thresh:
                        country_topics.append(i)

            country_topic_dict.update({country: country_topics})

        if debug:
            print("Country topic dict is")
            print(country_topic_dict)

        for setup in setups:
            setup_name = setup[0]

            aug_doc_file = os.path.join(config.AUG_DOC_META, 'doc_details_{}_aug_{}.pkl'.format('crisis', setup_name))
            aug_meta_df = pd.read_pickle(aug_doc_file)
            #aug_meta_df = aug_meta_df.filter(['country', 'country_n'])

            for doc_topic_min_level in document_topic_min_levels:

                save_name_append = ''

                if type(topic_f2_thresh) is tuple:
                    save_name_append += '_threshold_' + str(topic_f2_thresh[0]) + '_' + str(topic_f2_thresh[1])
                else:
                    save_name_append += '_threshold_' + str(topic_f2_thresh)
                print("Working on threshhold {}".format(topic_f2_thresh))

                save_name_append += '_setup_' + setup_name

                if type(doc_topic_min_level) is tuple:
                    save_name_append += '_docMinLevel_' + str(doc_topic_min_level[0]) + '_' + str(doc_topic_min_level[1])
                else:
                    save_name_append += '_docMinLevel_'+str(doc_topic_min_level)

                data_length = aug_meta_df.shape[0]

                if debug:
                    data_length = 400000  # Test

                new_aug_save_file = os.path.join(config.topiccing_aug_meta,
                                                 'doc_meta_aug{}.pkl'.format(save_name_append))
                new_aug_df = None  # Stores new information - account for the loss of countries for topic discrimination

                for part_i in range(num_of_series):
                    if debug:
                        if part_i == 2:
                            break  # Bail for testing

                    print("Working on part {}".format(part_i))
                    partition_start = part_i * partition_size
                    partition_end = min(partition_start + partition_size, data_length)

                    part_df = aug_meta_df[partition_start:partition_end]

                    if debug:
                        #test
                        print("PRE PART HEAD")
                        print(part_df.head())

                    this_series_file = series_base_file.format(part_i)
                    ds = pd.read_pickle(this_series_file)
                    part_df = part_df.join(ds, how="left")
                    del ds

                    # Discard zero-country documents
                    part_df = part_df[part_df['country_n'] > 0]

                    part_ind = list(part_df.index.values)  # Save index from dataframe
                    part_df['doc_topics'] = ""  # Fill doc_topics column with empty strings (alt - NaN)

                    # Identify topics expressed in each document
                    for this_document in part_ind:
                        doc_topics = []

                        # Look at top_n topics
                        if type(doc_topic_min_level) is tuple:
                            if doc_topic_min_level[0] == 'top':
                                top_n = doc_topic_min_level[1]
                                this_topics = list(part_df.at[this_document, '{}_predicted_topics'.format(model_name)])
                                this_topics.sort(key=lambda x: x[1], reverse=True)  # Sort descending order
                                just_topics = [x[0] for x in this_topics]  # TODO test

                                doc_topics = just_topics[:top_n]

                        # Look at all topics identified above a threshold
                        else:
                            all_topic = list(part_df.at[this_document, '{}_predicted_topics'.format(model_name)])
                            for i in range(num_topics):
                                if all_topic[i][1] >= doc_topic_min_level:
                                    doc_topics.append(i)
                        part_df.at[this_document, 'doc_topics'] = doc_topics

                    # Keep only countries with the proper topic-country matchup
                    for country in countries:  # Itterate over countried
                        valued_topics = country_topic_dict[country]
                        for this_document in part_ind:  # Iterate over filed in part
                            this_doc_countries = part_df.at[this_document, 'country']
                            if country in this_doc_countries:
                                # Save all countries, except this one
                                temp_countries = [x for x in this_doc_countries if x != country]
                                this_doc_topics = set(part_df.at[this_document, 'doc_topics'])

                                # If topics intersect, add this country back in
                                if len(this_doc_topics.intersection(valued_topics)) > 0:
                                    temp_countries.append(country)

                                # Override country info
                                part_df.at[this_document, 'country'] = temp_countries

                    # replace country_n info
                    part_df['country_n'] = part_df['country'].map(lambda x: len(x))

                    # Keep only documents and columns needed
                    part_df = part_df[part_df['country_n'] > 0]
                    part_df = part_df.drop(columns=['ldaviz_t100_predicted_topics', 'doc_topics'])

                    if new_aug_df is None:
                        new_aug_df = part_df
                    else:
                        new_aug_df = new_aug_df.append(part_df)

                new_aug_df.to_pickle(new_aug_save_file)

                print("New Aug DF saved at {}".format(new_aug_save_file))
