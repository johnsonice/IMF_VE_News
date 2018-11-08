import json
import os
import argparse
import gensim
from crisis_points import crisis_points
from evaluate import evaluate, get_recall, get_precision, get_fscore 
import pandas as pd

def list_topics(mod_path, topn=15):
    model = gensim.models.ldamodel.LdaModel.load(mod_path)
    topics = [[top[0] for top in model.show_topic(i, topn)] for i in range(model.num_topics)]
    return topics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mod_list', nargs='+', 
                        help='space delimited list of model paths')
    parser.add_argument('-topn', '--topn', action='store', dest='topn', type=int, default=15)
    parser.add_argument('-yp', '--years_prior', action='store', dest='years_prior', type=int, default=2)
    parser.add_argument('-w', '--window', action='store', dest='window', type=int, default=8)
    parser.add_argument('-c', '--countries', action='store', dest='countries', default=crisis_points.keys())
    parser.add_argument('-meth', '--method', action='store', dest='method', default='zscore')
    parser.add_argument('-cd', '--crisis_defs', action='store', dest='crisis_defs', default='kr')
    args = parser.parse_args()

    # Get prec, rec, and fscore for each country for each topic
    mods = args.mod_list
    for mod_path in mods:
        print('\nWorkig on {}'.format(mod_path))
        topics = list_topics(mod_path, args.topn)
        for i, topic in enumerate(topics): 
            print('\n\tworking on topic {}'.format(i))
            country_stats = []
            for country in args.countries:
                print('\r\t\tWorking on {}\t'.format(country), end='')
                stats = pd.Series(evaluate(topic, country, window=args.window, years_prior=args.years_prior,
                                           method=args.method, crisis_defs=args.crisis_defs),
                                  index=['recall','precision','fscore','tp','fp','fn'], name=country)
                country_stats.append(stats)
            all_stats = pd.DataFrame(country_stats)
            all_tp, all_fp, all_fn = all_stats['tp'].sum(), all_stats['fp'].sum(), all_stats['fn'].sum()
            all_recall = get_recall(all_tp, all_fn)
            all_prec = get_precision(all_tp, all_fp)
            all_f2 = get_fscore(all_tp, all_fp, all_fn, beta=2)
            aggregate_stats = pd.Series([all_recall, all_prec, all_f2, all_tp, all_fp, all_fn], 
                            name='aggregate', index=['recall','precision','fscore','tp','fp','fn'])
            summary = all_stats.append(aggregate_stats)

            # colate results
            results = {
                'topic': i,
                'terms': topic,
                'country_results': all_stats.to_dict(orient='index'),
                'overall_results': aggregate_stats.to_dict()
            }

            # Save summary
            save_dir = '../eval/{}'.format(os.path.basename(mod_path))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            summary.to_csv(os.path.join(save_dir, 'topic_{}_summary.csv'.format(i)))

            # Save results in json format
            with open(os.path.join(save_dir, 'topic_{}_results.json'.format(i)), 'w') as f:
                f.write(json.dumps(results))

            # Save terms in text format
            with open(os.path.join(save_dir, 'topic_{}_terms.txt'.format(i)), 'w') as f:
                for term in topic:
                    f.write(term + '\n')

