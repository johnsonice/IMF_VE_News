"""
For a given country, calculate word frequency for documents relating to the country and coming from crisis periods
and word freq for docs relating to the country and coming from non-crisis periods.
"""
from frequency_utils import list_crisis_docs, crisis_noncrisis_freqs
from crisis_points import crisis_points
import sys

if __name__ == '__main__':
    assert len(sys.argv) <= 2
    if len(sys.argv) == 2:
        countries = sys.argv[1] # calculate for list of countries supplied on command line, else for all
        assert isinstance(countries, list)
    else:
        countries = crisis_points.keys()

    for country in countries:
        print("\r\ncalculating freqs for {}".format(country), end=' ')
        crisis, non_crisis = list_crisis_docs(country)
        crisis_noncrisis_freqs(crisis, non_crisis, country, save=True)
