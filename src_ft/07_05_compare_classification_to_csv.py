import sys
sys.path.insert(0, './libs')
import argparse
import pandas as pd
import os
import config
import crisis_points

# TODO comments

combined_df = pd.DataFrame()
eval_types = [config.eval_type] #TODO TEMP
verbose = True


def add_file_to_df(pre_combined_df, read_file, names_dict):
    base_df = pd.read_csv(read_file)
    if verbose:
        print("Read evaluation df from", read_file)

    word_list = list(base_df['word'])
    new_df_dict = names_dict

    for j in range(len(word_list)):
        word = word_list[j]
        new_df_dict[word + '_recall'] = base_df['recall'][j]
        new_df_dict[word + '_prec'] = base_df['prec'][j]
        new_df_dict[word + '_f2'] = base_df['f2'][j]

    app_df = pd.DataFrame(new_df_dict)
    pre_combined_df = pre_combined_df.append(app_df)
    if verbose:
        print("Appended df", app_df)

    return pre_combined_df

for e_type in eval_types:
    for class_type_setup in config.class_type_setups:
        class_type = class_type_setup[0]

        if config.experiment_mode == 'country_classification':
            folder_path = os.path.join(config.EVAL_WG, class_type, e_type)
            file_path = os.path.join(folder_path, 'overall_agg_sim_True_overall_month_offset_{}_smoothwindow_'
                                                          '{}_evaluation.csv'.format(config.months_prior,
                                                                                     config.smooth_window_size))
            name_dict = {'classification_type': [class_type]}
            combined_df = add_file_to_df(combined_df, file_path, name_dict)

        elif config.experiment_mode == 'topiccing_discrimination':
            class_folder_path = os.path.join(config.topiccing_eval_wg, class_type)

            # Per each topic-power in-country
            for f2_thresh in config.topic_f2_thresholds:
                if type(f2_thresh) is tuple:
                    f2_thresh = '{}_{}'.format(f2_thresh[0], f2_thresh[1])
                else:
                    f2_thresh = str(f2_thresh)

                f2_folder_path = os.path.join(class_folder_path, f2_thresh)

                # Per each topic-level in-document
                for doc_thresh in config.document_topic_min_levels:
                    if type(doc_thresh) is tuple:
                        doc_thresh = '{}_{}'.format(doc_thresh[0], doc_thresh[1])
                    else:
                        doc_thresh = str(doc_thresh)

                    folder_path = os.path.join(f2_folder_path, doc_thresh)

                    if config.just_five:
                        folder_path = os.path.join(folder_path, 'j5_countries')

                    file_path = os.path.join(folder_path, 'overall_agg_sim_True_overall_month_offset_{}_smoothwindow_'
                                                          '{}_evaluation.csv'.format(config.months_prior,
                                                                                     config.smooth_window_size))

                    name_dict = {'classification_type': [class_type],
                                 'f2_threshold': [f2_thresh],
                                 'doc_topic_level': [doc_thresh]
                                 }

                    combined_df = add_file_to_df(combined_df, file_path, name_dict)

        # Only test assessment modes
        elif config.experiment_mode == "crisis_assessments":
            assess_dict = {
                'IMF_GAP_6': crisis_points.imf_gap_6_events,
                'IMF_GAP_0': crisis_points.imf_all_events,
                'LoDuca': crisis_points.crisis_points_LoDuca,
                'ReinhartRogoff': crisis_points.crisis_points_Reinhart_Rogoff_All,
                'RomerRomer': crisis_points.crisis_points_RomerNRomer,

            }
            assess_on = ['Min1_AllCountry', 'IMF_GAP_6', 'IMF_GAP_0', 'LoDuca', 'ReinhartRogoff', 'RomerRomer']

            for asses_type in assess_on:
                ev_path = os.path.join('/data/News_data_raw/FT_WD_research/eval/new_comp', asses_type)
                file_path = os.path.join(ev_path, 'overall_agg_sim_True_overall_month_offset_{}_smoothwindow_'
                                                  '{}_evaluation.csv'.format(config.months_prior,
                                                                             config.smooth_window_size))
                if asses_type == 'Min1_AllCountry':
                    ev_path = os.path.join('/data/News_data_raw/FT_WD_research/eval/new_comp', asses_type, e_type)

                name_dict = {'classification_type': [asses_type]}
                combined_df = add_file_to_df(combined_df, file_path, name_dict)

    # TODO modularize using args
    if config.experiment_mode == 'country_classification':
        out_file = os.path.join(config.EVAL_WG, 'classification_comparison',
                                'country_classification_comparison_using_{}.csv'.format(e_type))
    elif config.experiment_mode == 'topiccing_discrimination':
        out_file = os.path.join(config.topiccing_eval_wg, 'topiccing_comparison',
                                'topiccing_discrimination_comparison_using_{}.csv'.format(e_type))
    elif config.experiment_mode == "crisis_assessments":
        out_file = os.path.join('/data/News_data_raw/FT_WD_research/eval/new_comp/CrossSection/cross_comparison.csv')

    try:
        already_written = pd.read_csv(out_file)
        combined_df = already_written.append(combined_df)
        print("Adding to previous csv")
    except IOError:
        pass

    combined_df.set_index('classification_type')
    combined_df.to_csv(out_file, index=False)
    print("Saved dataframe in file {}".format(out_file))

