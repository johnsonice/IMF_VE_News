# -*- coding: utf-8 -*-
"""
python module for accessing Factiva's api
"""
import requests
import ujson as json
import os
from datetime import datetime as dt
import time
import random
from pandas import to_datetime


# =============================================================================
# Main Class
# =============================================================================
class FactivaAPI(object):

    def __init__(self, credentials_file):
        # ---- load credentials
        with open(credentials_file, 'r') as f:
            creds = json.loads(f.read())
            self.email = creds['Email']
            self.password = creds['Password']

        # ---- Setup session vars
        self.base_url = 'http://api.dowjones.com/api/1.0/'
        self.headers = {'Content-Type': 'application/xml'}
        self.sess_id = None
        self.logged_in = False

        # ---- log in
        self.login()

    def login(self):
        # ---- Generate data payload
        data = '''
        <SessionRequest>
            <Email>{}</Email>
            <Password>{}</Password>
            <Format>json</Format>
        </SessionRequest>
        '''.format(self.email, self.password)

        # ---- Hit the API
        login_url = self.base_url + 'session/'
        response = requests.post(login_url, data=data, headers=self.headers)

        # ---- Alert if login failed
        if not self._successful(response):
            print(response.text)
            raise Exception('Login Failed! check your login credentials')

        # ---- Set session vars
        sess_info = response.json()
        self.sess_id = sess_info['SessionId']
        self.logged_in = True
        print("sess_id: {}".format(self.sess_id))

    def logout(self, session_id=None):
        session_id = self.sess_id if session_id is None else session_id

        # ---- Query the API
        logout_url = self.base_url + 'session/JSON?SessionId={}'.format(session_id)
        response = requests.delete(logout_url)

        # ---- Raise Error if logout fails
        if not self._successful(response):
            print(response.text)
            raise Exception('Logout Failed!')

        # ---- Reset session ID and logged_in attributes
        self.sess_id = None
        self.logged_in = False

    def taxonomy(self, category):
        url = 'https://api.dowjones.com/taxonomy/factiva-{}'.format(category)
        headers = {'Accept': 'application/json', 'Authorization': '{}'.format(self.sess_id)}
        response = requests.get(url, headers=headers)
        return response

    def distributed_search(self, query_string, record_type='article', **kwargs):
        """
        For valid kwarg params, visit http://www.factiva.com/cp_developer/Producthelp/djif/webhelp/default.htm
        param record_type: what format to return the list of articles in.
            valid options: 'all', 'article', 'id'
        """
        assert record_type in ('all', 'article', 'id')

        # ---- Make sure user is logged in
        if not self.logged_in:
            raise Exception('You Aren\'t logged in!')

        # ---- Query the API with the base query string and any additional filter params
        offset = 0
        query_number = 1
        query_counter = 1
        records = kwargs['Records'] if 'Records' in kwargs else 100
        while query_counter > 0:
            # How many records to return for each query?
            nrecs = 100 if (records - offset >= 100) else (records - offset)

            # Send the HTTP Get
            search_url = self.base_url + 'Content/search/JSON?' + \
                         'QueryString={}'.format(query_string) + \
                         '&Offset={}&Records={}'.format(offset, nrecs) + \
                         ''.join(['&{}={}'.format(k, v) for k, v in kwargs.items() if k != 'Records']) + \
                         '&SessionId={}'.format(self.sess_id)
            response = requests.get(search_url)
            if response.status_code != 200:
                print('bad response code: {}'.format(response.status_code))
            response = response.json()

            # Get starting n_queries
            if query_number == 1:
                records = response['TotalRecords']
                total_queries = query_counter = (records // 100) if (records % 100) == 0 else (records // 100) + 1

            # Sleep
            if query_number % 21 == 0 and total_queries > 20:
                self.logout()
                delay = 301
                self.login()
            else:
                delay = random.randint(0, 9) * 0.5
            time.sleep(delay)

            # Print status
            print("query {} out of {}".format(query_number, total_queries))

            # Yield content
            try:
                if record_type == 'all':
                    yield response
                elif record_type in ('article', 'id'):
                    for article in response['Articles']:
                        if record_type == 'article':
                            yield article
                        else:
                            yield article['ArticleId']

                offset += nrecs
                query_counter -= 1
                query_number += 1

            except KeyError:
                print('No Articles in this query response!\nsleeping...')
                time.sleep(301)


    def search(self, query_string, **kwargs):
        # Send the HTTP Get
        search_url = self.base_url + 'Content/search/JSON?' + \
                     'QueryString={}'.format(query_string) + \
                     ''.join(['&{}={}'.format(k, v) for k, v in kwargs.items()]) + \
                     '&SessionId={}'.format(self.sess_id)
        response = requests.get(search_url).json()
        return response

    def get_article(self, ID, parts='Body'):
        """
        param parts: list of article parts (see http://www.factiva.com/cp_developer/Producthelp/djif/webhelp/appendix/article_display_parts.htm)
        param ID: article id of list of art ids.
        """
        # NOTE: Apparently the API has a max retrieval of 137 docs at a time (this is not
        # Documented anywhere, nor does there seem to be a good reason for it...)

        # ---- Chunk IDs into lists of length 137
        chunks = self._chunk(ID, 137) if isinstance(ID, list) else [[ID]]
        # ---- Handle lists vs. strings in input
        parts_string = "|".join(parts) if isinstance(parts, list) else parts
        results = []
        for chunk in chunks:
            id_string = "|".join(chunk)
            # ---- Construct URL query
            get_art_url = self.base_url + 'content/article/JSON?Id={}'.format(id_string) + \
                          '&parts={}'.format(parts_string) + \
                          '&SessionId={}'.format(self.sess_id)

            # ---- process and return results
            response = requests.get(get_art_url)
            if not self._successful(response):
                raise Exception('API call to get articles failed with response {}!'.format(response.status_code))
            try:
                results.append(response.json()['Articles'])
            except KeyError:
                results.append(response.json())

        results = [art for batch in results for art in batch]
        return results

    def id2text(self, ID):
        """
        Return just the text from the body of the articles specified
        """
        # TODO: figure out how the named entities in the body are tagged.
        # ---- Get Articles
        articles = self.get_article(ID, parts='body')
        texts = [self.article2text(art) for art in articles]
        return texts

    @staticmethod
    def article2text(article):
        # ---- Make sure the article actually has text to pull out
        try:
            paras = [i['PItems'] for i in article['Body']]
        except KeyError as e:
            id = article['ArticleId']
            print('article {} does not contain a body section -- skipping'.format(id))
            return False

        # ---- Parse sentences
        sents = []
        for p in paras:
            for sent in p:
                # Pull out normal text
                if 'Value' in sent.keys():
                    sents.append(sent['Value'])
                # Pull out named entities (Not sure how these are tagged yet)
                elif 'Name' in sent.keys():
                    sents.append('<<{}>>'.format(sent['Name']))
                else:
                    continue

        # ---- return text
        return ' '.join(sents)

    @staticmethod
    def publication_date(article):
        timestamp = article['PubDateTime']
        timestamp = timestamp.replace('/Date(', '')
        timestamp = timestamp.replace(')/', '')
        time = int(timestamp)
        date = to_datetime(dt.fromtimestamp(time / 1e3))
        return date

    def create_corpus(self, query_string, output_dir, record_type='article', **kwargs):
        """
        creates a corpus in the specified output dir using the specified **kwargs
        and query string to perform an article search. Will write each article
        to it's own file within the output directory.

        param record_type: what record to write to file for each article.
            valid types: 'article' 'text' or 'id'
        """
        assert record_type in ('article', 'text', 'id')  # ensure record type

        # ---- Get article id list for search query
        art_stubs = self.search(query_string, record_type='article', **kwargs)
        ids = [art['ArticleId'] for art in art_stubs['Articles']]

        # ---- Write Ids to file
        if record_type == 'id':
            outf = os.path.join(output_dir, '{}_article_ids.txt'.format(query_string))
            with open(outf, 'w') as f:
                for i in ids:
                    f.write(i + '\n')
            return

        # ---- Dump JSON articles or extracted text to file
        if (record_type == 'article') or (record_type == 'text'):
            articles = self.get_article(ids)
            ext = 'json' if record_type == 'article' else 'txt'
            for art in articles:
                outf = os.path.join(output_dir, '{}.{}'.format(art['ArticleId'], ext))
                with open(outf, 'w', encoding='utf8') as f:
                    if record_type == 'article':
                        f.write(json.dumps(art, indent=2))
                    else:
                        f.write(self.article2text(art))
            return

    @staticmethod
    def _successful(response):
        if not response.status_code == 200:
            return False
        else:
            return True

    def _chunk(self, lst, n):
        """
        Chunk a list into sublists of length n
        """
        for i in range(0, len(lst), n):
            yield lst[i: i + n]


def _emergency_log_out(sess_id):
    base_url = 'http://api.dowjones.com/api/1.0/'
    logout_url = base_url + 'session/JSON?SessionId={}'.format(sess_id)
    response = requests.delete(logout_url)

    # ---- Raise Error if logout fails
    if not FactivaAPI._successful(response):
        print(response.text)
        raise Exception('Logout Failed!')


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    creds = 'credentials.json'
    api = FactivaAPI(creds)
    try:
        x = api.search('gabon')
        print(json.dumps(x, indent=2))
    except Exception as e:
        raise e
    finally:
        api.logout()
