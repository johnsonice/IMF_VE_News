import pandas as pd
import numpy as np


def read_grouped_search_words(file_path):
    """
    Read the search words, by group, from the indicated file
    :param file_path:
    :return: 'dict' mapping key(group name) -> value(list of str words-in-group)
    """

    df = pd.read_csv(file_path)
    search_groups = df.to_dict()
    for k, v in search_groups.items():
        temp_list = [i for i in list(v.values()) if not pd.isna(i)]  # Save all non-NA values - different len groups
        temp_list = [wg.split('&') for wg in temp_list]   # split & for wv search
        search_groups[k] = temp_list  # Map str key group name -> list[str] group words
    return search_groups


topic_df = pd.read_csv('../../csv_out/sorted_lda_100_topic_words.csv', index_col=0)
search_dict = read_grouped_search_words('../../research/grouped_search_words_final.csv')

# TODO lemmatize both lists, count matching lemmas
# TODO generate the associated words from w2v and count those intersects

# TODO RANDOM OTHER THING FOR MEETING - Graphs of ROCs, Graphs of Non-w2v sentiment v. w2v sentiment (both levels,
#   and roc?)

# Count intersection of group-words seed words and topic top 30 words
for this_word_group in list(search_dict.keys()):
    topic_df[this_word_group] = np.nan
    for top in range(100):
        topic_words = set(topic_df.loc[top])
        search_words = search_dict[this_word_group]
        print(search_words)
        intersect = topic_words.intersection(search_words)
        topic_df.loc[top, this_word_group] = len(intersect)
    print(topic_df)  ## TEMP


