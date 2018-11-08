import argparse
import gensim
import re

def print_topics(model_name):
    model = gensim.models.ldamodel.LdaModel.load(model_name)
    topics = model.print_topics(-1)
    with open(model_name + "_topics", 'w') as f:
        for i,top in topics:
            top = re.sub("\d+\.\d+\*", "", top)
            top = re.sub(" + ", ", ", top)
            top = re.sub('"', '', top)
            f.write('\n{}: {}'.format(i, top))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', action='store', dest='models', required=True)
    args = parser.parse_args()

    models = args.models.split(',')
    for m in models:
        print_topics(m)
    

