from load_ll_crisis_util import get_ll_crisis_points


crisis_points = {
    'argentina': {
        'starts': ['1980-03', '1985-05', '1994-12', '2001-12'],
        'peaks': ['1982-07', '1989-06', '1995-03', '2001-12'],
        'bop': ['1981-02', '1986-09', '1990-02']
    },
    'bolivia': {
        'starts': ['1987-10'],
        'peaks': ['1988-06'],
        'bop': ['1985-09']
    },
    'brazil': {
        'starts': ['1985-11', '1994-12', '1999-01'],
        'peaks': ['1985-11', '1996-03', '1999-01'],
        'bop': ['1986-11', '1991-10']
    },
    'chile': {
        'starts': ['1981-09'],
        'peaks': ['1983-03'],
        'bop': ['1982-08']
    },
    'colombia': {
        'starts': ['1982-07'],
        'peaks': ['1985-06'],
        'bop': ['1983-03']
    },
    'denmark': {
        'starts': ['1987-03'],
        'peaks': ['1990-06'],
        'bop': ['1983-08']
    },
    'finland': {
        'starts': ['1991-09'],
        'peaks': ['1992-06'],
        'bop': ['1991-11']
    },
    'indonesia': {
        'starts': ['1992-11'],
        'peaks': ['1992-11'],
        'bop': ['1986-09']
    },
    'israel': {
        'starts': ['1983-10'],
        'peaks': ['1984-06'],
        'bop': ['1983-10']
    },
    'malaysia': {
        'starts': ['1985-07'],
        'peaks': ['1986-08'],
        'bop': ['1975-07']
    },
    'mexico': {
        'starts': ['1982-09', '1992-10'],
        'peaks': ['1984-06', '1996-03'],
        'bop': ['1982-12', '1994-12']
    },
    'norway': {
        'starts': ['1988-11'],
        'peaks': ['1991-10'],
        'bop': ['1986-05']
    },
    'peru': {
        'starts': ['1983-03'],
        'peaks': ['1983-04'],
        'bop': ['1987-04']
    },
    'philippines': {
        'starts': ['1981-01'],
        'peaks': ['1985-06'],
        'bop': ['1983-10']
    }, 
    'spain': {
        'starts': ['1978-11'],
        'peaks': ['1983-01'],
        'bop': []
    },
    'sweden': {
        'starts': ['1991-11'],
        'peaks': ['1992-09'],
        'bop': ['1992-11']
    },
    'thailand': {
        'starts': ['1979-03', '1983-10'],
        'peaks': ['1979-03', '1985-06'],
        'bop': ['1978-11', '1984-11']
    },
    'turkey': {
        'starts': ['1991-01'],
        'peaks': ['1991-03'],
        'bop': ['1994-03']
    },
    'uruguay': {
        'starts': ['1981-03'],
        'peaks': ['1985-06'],
        'bop': ['1982-10']
    },
    'venezuela': {
        'starts': ['1993-10'],
        'peaks': ['1994-08'],
        'bop': ['1994-05']
    },
}
crisis_points_lo_duca = {
    'austria': {
        'starts': ['2007-12', ],
        'peaks': ['2016-04', '1985-06'],
    },
    'belgium': {
        'starts': ['2007-11'],
        'peaks': ['2012-12'],
    },
    'bulgaria': {
        'starts': ['1996-05'],
        'peaks': ['1997-07'],
    },
    'cyprus': {
        'starts': ['2000-01', '2011-06'],
        'peaks': ['2001-03', '2016-03'],
    },
    'czechia': {
        'starts': ['1997-05'],
        'peaks': ['2000-06'],
    },
    'germany': {
        'starts': ['1974-06', '2001-01', '2007-08'],
        'peaks': ['1974-11', '2003-11', '2013-06'],
    },
    'denmark': {
        'starts': ['1987-03', '2008-01'],
        'peaks': ['1995-01', '2013-12'],
    },
    'estonia': {
        'starts': ['1992-11', '1994-08', '1998-06'],
        'peaks': ['1993-03', '1994-09', '1998-10'],
    },
    'spain': {
        'starts': ['1978-01', '2009-03'],
        'peaks': ['1985-09', '2013-12'],
    },
    'finland': {
        'starts': ['1991-09'],
        'peaks': ['1996-12'],
    },
    'france': {
        'starts': ['1991-06', '2008-04'],
        'peaks': ['1995-03', '2009-11'],
    },
    'greece': {
        'starts': ['2010-05'],
        'peaks': ['9999-12'], # ?? marked as 'ongoing'
    },
    'croatia': {
        'starts': ['1998-04', '2007-09'],
        'peaks': ['2000-01', '2012-06'],
    },
    'hungary': {
        'starts': ['1991-01', '2008-09'],
        'peaks': ['1995-12', '2010-08'],
    },
    'ireland': {
        'starts': ['2008-09'],
        'peaks': ['2013-12'],
    },
    'italy': {
        'starts': ['1991-09', '2011-08'],
        'peaks': ['1997-12', '2013-12'],
    },
    'lithuania': {
        'starts': ['1995-01', '2008-12'],
        'peaks': ['1996-12', '2009-11'],
    },
    'luxembourg': {
        'starts': ['2008-01'],
        'peaks': ['2010-10'],
    },
    'latvia': {
        'starts': ['1995-05', '2008-11'],
        'peaks': ['1996-06', '2010-08'],
    },
    'netherlands': {
        'starts': ['2008-01'],
        'peaks': ['2013-02'],
    },
    'norway': {
        'starts': ['1988-09', '2008-09'],
        'peaks': ['1993-11', '2009-10'],
    },
    'poland': {
        'starts': ['1981-03', '1992-01'],
        'peaks': ['1994-10', '1994-12'],
    },
    'portugal': {
        'starts': ['1983-02', '2008-10'],
        'peaks': ['1985-03', '2015-12'],
    },
    'romania': {
        'starts': ['1981-11', '1996-01', '2007-11'],
        'peaks': ['1989-12', '2000-12', '2010-08'],
    },
    'sweden': {
        'starts': ['1991-01', '2008-09'],
        'peaks': ['1997-06', '2010-10'],
    },
    'slovenia': {
        'starts': ['1991-06', '2009-12'],
        'peaks': ['1994-07', '2014-12'],
    },
    'united-kingdom': {
        'starts': ['1973-11', '1991-07', '2007-08'],
        'peaks': ['1975-12', '1994-04', '2010-01'],
    }
}
country_dict_all = {
    'argentina': ['argentina'],
    'austria': ['austria'],
    'bolivia': ['bolivia'],
    'brazil': ['brazil'],
    'chile': ['chile'],
    'colombia': ['colombia'],
    'croatia': ['croatia'],
    'cyprus': ['cyprus'],
    'czechia': ['czechia', 'prague', 'czech', 'czech-republic'],
    'denmark': ['denmark'],
    'finland': ['finland'],
    'indonesia': ['indonesia'],
    'israel': ['israel'],
    'south-korea':['south korea','korean','south-korea','south-korean','seoul'],
    'malaysia': ['malaysia'],
    'mexico': ['mexico'],
    'norway': ['norway'],
    'peru': ['peru'],
    'philippines': ['philippines'],
    'slovenia': ['slovenia'],
    'spain': ['spain'],
    'sweden': ['sweden'],
    'thailand': ['thailand'],
    'turkey': ['turkey'],
    'uruguay': ['uruguay'],
    'venezuela': ['venezuela'], ## these is the original list
    'australia': ['Australia','Australian','Sydney'],
    'belgium': ['Belgium','Belgien','Brussels'],
    'bulgaria': ['Bulgaria','Bulgarian','Sofia'],
    'canada': ['Canada','Canadian','Toronto'],
    'china': ['China','Chinese','Beijing'],
    'ecuador': ['Ecuador','Ecuadorian','Quito'],
    'egypt': ['Egypt','Egyptian','Cairo'],
    'france': ['France','French','Paris'],
    'germany': ['Germany','German','Berlin'],
    'greece': ['Greece','Greek','Athens'],
    'hungary': ['Hungary','Hungarian','Budapest'],
    'iceland': ['Iceland','Icelander','Reykjavik'],
    'india': ['India','Indian','Mumbai'],
    'ireland': ['Ireland','Irish','Dublin'],
    'italy': ['Italy','Italian','Rome'],
    'jamaica': ['Jamaica','Jamaican'],
    'japan': ['Japan','Japanese','Tokyo'],
    'jordan': ['Jordan','Jordanian','Amman'],
    'kenya': ['Kenya','Kenyan','Nirobi'],
    'latvia': ['Latvia','Latvian','Riga'],
    'lebanon': ['Lebanon','Lebanese','Beirut'],
    'netherlands': ['Netherlands','Dutch','Amsterdam'],
    'new-zealand': ['New Zealand','New Zealander','Auckland'],
    'nicaragua': ['Nicaragua','Nicaraguan','Managua'],
    'nigeria': ['Nigeria','Nigerian','Lagos'],
    'pakistan': ['Pakistan','Pakistani','Karachi'],
    'poland': ['Poland','Polish','Warsaw'],
    'russia': ['Russia','Russian','Moscow'],
    'south-africa': ['South Africa','South African','Johannesburg'],
    'tanzania': ['Tanzania','Tanzanian','Dar es Salaam'],
    'tunisia': ['Tunisia','Tunisian','Tunis'],
    'uganda': ['Uganda','Ugandan','Kampala'],
    'ukraine': ['Ukraine','Ukrainian','Kiev'],
    'united-kingdom': ['United Kingdom','British','London','UK'],
    'united-states': ['US','United States','American','America','Washington','united states','washington'],
    'vietnam': ['Vietnam','Vietnamese ','Hanoi','Viet Nam'],
    'zambia': ['Zambia','Lusaka'],
    'zimbabwe': ['Zimbabwe','Zimbabwean','Harrare'],
    'lithuania': ['lithuania'],
    'luxembourg': ['luxembourg'],
    'portugal': ['portugal'],
    'romania': ['romania'],
}

countries_lo_duca = list(crisis_points_lo_duca.keys())


country_dict_original = {
    'argentina': ['argentina'],
    'bolivia': ['bolivia'],
    'brazil': ['brazil'],
    'chile': ['chile'],
    'colombia': ['colombia'],
    'denmark': ['denmark'],
    'finland': ['finland'],
    'indonesia': ['indonesia'],
    'israel': ['israel'],
    'malaysia': ['malaysia'],
    'mexico': ['mexico'],
    'norway': ['norway'],
    'peru': ['peru'],
    'philippines': ['philippines'],
    'spain': ['spain'],
    'sweden': ['sweden'],
    'thailand': ['thailand'],
    'turkey': ['turkey'],
    'uruguay': ['uruguay'],
    'venezuela': ['venezuela'], ## these is the original list
}

country_dict = country_dict_original

import os 
cd = os.path.dirname(os.path.abspath(__file__))
ll_crisis_points=get_ll_crisis_points(os.path.join(cd ,'ll_crisis_dates.xlsx'),'import',list(country_dict.keys()))


#%%
if __name__ == '__main__':
    import ujson as json
    with open('crisis_points.json', 'w') as f:
        f.write(json.dumps(crisis_points))
    
