import re

# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
def get_country_name(text, country_dict, rex=None):
    for c, v in country_dict.items():
        if c in ['united-states']:
            rex = construct_rex(v, case=True)
        else:
            rex = construct_rex(v)
        rc = rex.findall(text)
        if len(rc) > 0:
            yield c

def get_country_name_count(text, country_dict, min_count=0, rex=None):
    for c, v in country_dict.items():
        if c in ['united-states']:
            rex = construct_rex(v, case=True)
        else:
            rex = construct_rex(v)
        rc = rex.findall(text)
        l_rc = len(rc)
        if l_rc > 0 and l_rc >= min_count:
            yield [c, l_rc]


def construct_rex(keywords, case=False):
    r_keywords = [r'\b' + re.escape(k) + r'(s|es|\'s)?\b' for k in keywords]
    if case:
        rex = re.compile('|'.join(r_keywords))  # --- use case sentitive for matching for cases lik US
    else:
        rex = re.compile('|'.join(r_keywords), flags=re.I)  ## ignore casing
    return rex


def get_countries(article, country_dicts):
    # snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    # title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        # title.extend(snip)
        title = "{} {}".format(title, snip)
        cl = list(get_country_name(title, country_dict))
    elif title:
        cl = list(get_country_name(title, country_dict))
    elif snip:
        cl = list(get_country_name(snip, country_dict))
    else:
        cl = list()

    return article['an'], cl

def get_countries_by_count(article, country_dicts, min_this, max_other=None):
    # snip = word_tokenize(article['snippet'].lower()) if article['snippet'] else None
    # title = word_tokenize(article['title'].lower()) if article['title'] else None
    snip = article['snippet'].lower() if article['snippet'] else None
    title = article['title'].lower() if article['title'] else None
    if snip and title:
        # title.extend(snip)
        title = "{} {}".format(title, snip)
        cl = list(get_country_name_count(title, country_dict, min_this))
    elif title:
        cl = list(get_country_name_count(title, country_dict, min_this))
    elif snip:
        cl = list(get_country_name_count(snip, country_dict, min_this))
    else:
        cl = list()

    # get just first col
    cl = list(list(zip(*cl))[0])

    return article['an'], cl

article = {
    'an': 'article1',
    'snippet': 'snip snip snip snip snip snip argentina chile colombia denmark chile chile',
    'title': 'argentina bolivia chile title time lol argentina'
}


country_dict = {
    'argentina': ['argentina'],
    'bolivia': ['bolivia'],
    'brazil': ['brazil'],
    'chile': ['chile'],
    'colombia': ['colombia'],
    'denmark': ['denmark']
}

print("YIELD TEST : : :")
yield_test = get_countries(article, country_dict)
print(yield_test)
print("RETURN TEST : : :")
return_test = get_countries(article, country_dict)
print(return_test)
print("COUNTING TEST : : :")
min_this_country = 3
return_test = get_countries_by_count(article, country_dict, min_this_country)
print(return_test)
print(return_test[1])