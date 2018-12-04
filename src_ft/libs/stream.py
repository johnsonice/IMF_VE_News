from abc import ABC, abstractmethod
import os
from glob import glob
from collections import Iterable
import itertools
import ujson as json
from string import punctuation as punct

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import gensim
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stops
from multiprocessing import Pool
import time 

#---- Base Classes
class Streamer(ABC):
    """
    abstract base class for Generator that yields info from each doc in a dir
    :param input: File or Dir
    """
    def __init__(self, input, language=None, max_docs=None, phraser=None, regions=[],
                 region_inclusive=False, title_filter=None, verbose=False, 
                 stopwords=stops, tagged=True, parsed=False, ner=False, lemmatize=False, model="en_core_web_lg"):
        self.input = input
        self.input_files = self.get_input_files()
        self.language = language
        self.max_docs = max_docs
        self.phraser = gensim.models.phrases.Phraser.load(phraser) if type(phraser) == str else phraser
        self.regions = regions
        self.region_inclusive = region_inclusive
        self.title_filter = title_filter
        self.iteration = 1
        self.verbose = verbose
        self.stopwords = stopwords
        self.model = model
        self.nlp = None
        self.tagged = tagged
        self.parsed = parsed
        self.ner = ner
        self.lemmatize = lemmatize
        self.total_docs = 0
        self.stemmer = SnowballStemmer('english')

        assert language in ('en', 'fr', 'sp', None)
        assert type(phraser) in (type(None), gensim.models.phrases.Phraser, str)

    def head(self, n=10):
        return list(itertools.islice(self,  n))
    
    
    
    def process_json(self,f):
        try:
            with open(f, 'r', encoding="utf-8") as f:
                data = json.loads(f.read())
        except:
            print('File not found:{}'.format(f))
            return None
        
        if self.language and data['language_code'] != self.language:
                return None
                    # Region Filter:
        region_matched = 0
        if self.regions:
            region_codes = set(data['region_codes'].split(','))
            if not any(region_codes.intersection(set(self.regions))):
                   if not self.region_inclusive:
                        return None
            else:
                region_matched = 1
        
        if self.title_filter and not region_matched:
            snip = word_tokenize(data['snippet'].lower()) if data['snippet'] else None
            title = word_tokenize(data['title'].lower()) if data['title'] else None

            title_flag = True if title and any([tok.lower() in title for tok in self.title_filter]) else False
            snip_flag = True if snip and any([tok.lower() in snip for tok in self.title_filter]) else False

            if not any([title_flag, snip_flag]):
                return None
        # Retreive and yield output
        output = self.retrieve_output(data)
        return output
    
    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    def multi_process_files(self,workers=os.cpu_count()-1,chunk_size=1000):
        print('Start multiprocessing {} files in {} cores'.format(len(self.input_files),workers))
        start = time.time()
        batch_size = workers*chunk_size*5
        batches = list(self.chunks(self.input_files, batch_size))
        p = Pool(workers)
        
        res = list()
        for i in range(len(batches)):
            print('Processing {} - {} files ...'.format(i*batch_size,(i+1)*batch_size))
            rs = p.map(self.process_json, batches[i],chunk_size)
            res.extend(rs)
        p.close()
        p.join()
        end = time.time()
        
        ## filter None items
        res = [r for r in res if r is not None]
        results=[]
        for r in res:
            results.extend(r)
            
        print(time.strftime('%H:%M:%S', time.gmtime(end - start)))

        return results
    
    
    def get_input_files(self):
        # Check that valid dir or file or list of files is supplied
        if isinstance(self.input, str) and os.path.isdir(self.input):
            flist = glob(self.input + "/*.json")
        elif isinstance(self.input, str) and os.path.isfile(self.input):
            flist = [self.input]
        elif isinstance(self.input, Iterable):
            flist = self.input
        else:
            raise ValueError('INVALID FILE OR LIST OF FILES OR DIRECTORY PATH')
        return flist
    
    def __iter__(self):
        self.total_docs = 0
        
#        # Check that valid dir or file or list of files is supplied
#        if isinstance(self.input, str) and os.path.isdir(self.input):
#            flist = glob(self.input + "/*.json")
#        elif isinstance(self.input, str) and os.path.isfile(self.input):
#            flist = [self.input]
#        elif isinstance(self.input, Iterable):
#            flist = self.input
#        else:
#            raise ValueError('INVALID FILE OR LIST OF FILES OR DIRECTORY PATH')
        flist = self.input_files
        flist_length = len(flist)

        if self.verbose:
            print('Total number of documents: {}'.format(flist_length))
            print("\niteration: {}".format(self.iteration))
            self.iteration += 1

        # Iterate over all docs in the corpus, process and serve them if they pass the filters.
        for i, f in enumerate(flist):
            # Stop after max docs reached
            if self.max_docs and self.total_docs >= self.max_docs:
                break

            # Load the data
            try:
                with open(f, 'r', encoding="utf-8") as f:
                    data = json.loads(f.read())
                    if self.verbose:
                        print("\rProcessing " + str(i) + " of " + str(flist_length), end='',flush=True)
                    text = data['body']
            except:
                print('File not found:{}'.format(f))
                continue
        

            # Language Filter
            if self.language and data['language_code'] != self.language:
                continue

            # Region Filter:
            region_matched = 0
            if self.regions:
                region_codes = set(data['region_codes'].split(','))
                if not any(region_codes.intersection(set(self.regions))):
                    if not self.region_inclusive:
                        continue
                else:
                    region_matched = 1

            # Title Filter
            if self.title_filter and not region_matched:
                snip = word_tokenize(data['snippet'].lower()) if data['snippet'] else None
                title = word_tokenize(data['title'].lower()) if data['title'] else None

                title_flag = True if title and any([tok.lower() in title for tok in self.title_filter]) else False
                snip_flag = True if snip and any([tok.lower() in snip for tok in self.title_filter]) else False

                if not any([title_flag, snip_flag]):
                    continue

            # Retreive and yield output
            output = self.retrieve_output(data)
            self.total_docs += 1
            for item in output:
                yield item

    @abstractmethod
    def retrieve_output(self, data):
        ...


class SpacyStreamer(Streamer):
    def __init__(self, input, language=None, max_docs=None, phraser=None, regions=[],
                 region_inclusive=False, title_filter=None, verbose=False, 
                 stopwords=stops, tagged=True, parsed=False, ner=False,
                 lemmatize=True, model="en_core_web_lg"):
        super().__init__(input, language, max_docs, phraser, regions, region_inclusive, title_filter, 
                         verbose, stopwords, tagged, parsed, ner, lemmatize, model)
 
        self.nlp = spacy.load(self.model)

        # Set up spacy pipeline
        if self.stopwords:
            for w in self.stopwords:
                self.nlp.vocab[w].is_stop = True
        if not self.tagged:
            self.nlp.remove_pipe('tagger')
        if not self.ner:
            self.nlp.remove_pipe("ner")
        if not self.parsed:
            self.nlp.remove_pipe("parser")
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))


#---- Spacy Child classes
class SentStreamer(SpacyStreamer):
    def retrieve_output(self, data):
        text = data['body']
        doc = self.nlp(text)
        # remove stops and non-alphas
        if self.stopwords:
            doc = [[tok for tok in sent if tok.is_alpha and not tok.is_stop] for sent in doc.sents]

        # Lemmatize 
        if self.lemmatize:
            doc = [[tok.lemma_ for tok in sent] for sent in doc]
        else:
            doc = [[tok.text for tok in sent] for sent in doc]

        # Phrasegrams
        if self.phraser:
            doc = [self.phraser[sent] for sent in doc]

        return doc


class DocStreamer(SpacyStreamer):
    def retrieve_output(self, data):
        text = data['body']
        doc = self.nlp(text)
        # Remove stops and non-alphas
        if self.stopwords:
            doc = [t for t in doc if t.is_alpha and not t.is_stop] if self.stopwords else doc

        # Lemmatize
        if self.lemmatize:
            doc = [tok.lemma_ for tok in doc]
        else:
            doc = [tok.text for tok in doc]


        # Phrasegrams
        if self.phraser:
            doc = self.phraser[doc]


        output = [doc]
        return output


class FileStreamer(SpacyStreamer):
    def retrieve_output(self, data):
        text = data['body']
        doc = self.nlp(text)
        data['body'] = doc
        return [data]


#---- Fast classes
class SentStreamer_fast(Streamer):
    def retrieve_output(self, data):
        text = data['body']
        sents = [[tok for tok in word_tokenize(sent) if tok not in self.stopwords and tok not in punct] for sent in sent_tokenize(text)]
        if self.phraser:
            sents = [self.phraser[sent] for sent in sents]
        if self.lemmatize:
            sents = [[self.stemmer.stem(tok) for tok in sent] for sent in sents]
        return sents


class DocStreamer_fast(Streamer):
    def retrieve_output(self, data):
        text = word_tokenize(data['body'])
        text = [tok for tok in text if tok not in self.stopwords and tok not in punct]
        if self.phraser:
            text = self.phraser[text]
        if self.lemmatize:
            text = [self.stemmer.stem(tok) for tok in text] 
        return [text]


class FileStreamer_fast(Streamer):
    def retrieve_output(self, data):
        text = data['body']
        doc = word_tokenize(text)
        data['body'] = doc
        return [data]

class MetaStreamer_fast(Streamer):
    def retrieve_output(self, data):
        data['body'] = None
        return [data]
