## install some packages 
conda install -c conda-forge spacy glob
conda install gensim nltk numpy pandas jupyter notebook spyder glob2 scikit-learn seaborn
## install language module 
python -m spacy download en_core_web_lg

## all tho following are automatic now - no need to run 
## pip install .tar.gz archive from path or URL
#pip install /Users/you/en_core_web_sm-2.0.0.tar.gz
## set up shortcut link to load installed package as "en_default"
#python -m spacy link en_core_web_sm en_default


