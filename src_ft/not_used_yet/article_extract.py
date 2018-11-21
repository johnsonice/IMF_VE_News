"""
For a given time slice, get monthly frequency of articles about a particular 
country mentioning particular key words (from Factiva database)

Note: This was used to approx. the number of relative articles in Factiva. No longer needed or useful.
"""

from factiva import FactivaAPI
from datetime import datetime as dt
import pandas as pd
import ujson as json
import re


# ---- Login
creds = 'credentials.json'
api = FactivaAPI(creds)
with open('sessid.txt', 'w') as f:
    f.write(api.sess_id)

# ---- Constants
COUNTRY = 'argentina'
START_YEAR = 1995
END_YEAR = 1996
KEY_WORDS = " or ".join("debt, crisis, financial risk, economic risk, vulnerabilities, deficit, " \
            "unsustainable, recession, market instability, financial instability, economic instability, "\
            "external vulnerabilities, market instabilities, financial instabilities, economic instabilities".split(", "))

# ---- Build search generator
articles = api.distributed_search("(sc=j or sc=ftfta) and {} and ({})".format(COUNTRY, KEY_WORDS),
                                  Records=99999999, record_type='article', StartDate="01/01/{}".format(START_YEAR),
                                  EndDate="12/30/{}".format(END_YEAR),
                                  SearchMode="Traditional")

# ---- Shuttle article dates into dataframe
times = [(art['PubDateTime'].replace("/Date(","").replace(")/",""), art['ArticleId']) for art in articles]
time_df = pd.DataFrame([x[1] for x in times], index=[pd.to_datetime(dt.fromtimestamp(int(x[0]) / 1e3)) for x in times])
time_df['month'] = time_df.index.to_period('M')
month_counts = time_df.groupby('month').count()
time_df.to_json("corpus/KEYS_{}_{}_{}.json".format(COUNTRY, START_YEAR, END_YEAR))

# ---- Logout
api.logout()
