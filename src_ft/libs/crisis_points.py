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

country_dict_all = {
    #'argentina': ['argentina'],
    #'bolivia': ['bolivia'],
    #'brazil': ['brazil'],
    #'chile': ['chile'],
    #'colombia': ['colombia'],
    #'denmark': ['denmark'],
    #'finland': ['finland'],
    #'indonesia': ['indonesia'],
    #'israel': ['israel'],
    #'south-korea':['south korea','korean','south-korea','south-korean','seoul'],
    #'malaysia': ['malaysia'],
    #'mexico': ['mexico'],
    #'norway': ['norway'],
    #'peru': ['peru'],
    #'philippines': ['philippines'],
    #'spain': ['spain'],
    #'sweden': ['sweden'],
    #'thailand': ['thailand'],
    #'turkey': ['turkey'],
    #'uruguay': ['uruguay'],
    #'venezuela': ['venezuela'], ## these is the original list
    #'australia': ['Australia','Australian','Sydney'],
    #'belgium': ['Belgium','Belgien','Brussels'],
    #'bulgaria': ['Bulgaria','Bulgarian','Sofia'],
    #'canada': ['Canada','Canadian','Toronto'],
    #'china': ['China','Chinese','Beijing'],
    #'ecuador': ['Ecuador','Ecuadorian','Quito'],
    #'egypt': ['Egypt','Egyptian','Cairo'],
    #'france': ['France','French','Paris'],
    #'germany': ['Germany','German','Berlin'],
    #'greece': ['Greece','Greek','Athens'],
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
}

country_dict_temp = {
    'argentina': ['argentina'],
    'bolivia': ['bolivia'],
    'brazil': ['brazil'],
    'chile': ['chile'],
    'colombia': ['colombia'],
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
}

country_dict = country_dict_temp



#country_dict = {
#    'argentina': ['argentina'],
#    'bolivia': ['bolivia'],
#    'brazil': ['brazil'],
#    'chile': ['chile'],
#    'colombia': ['colombia'],
#    'denmark': ['denmark'],
#    'finland': ['finland'],
#    'indonesia': ['indonesia'],
#    'israel': ['israel'],
#    'malaysia': ['malaysia'],
#    'mexico': ['mexico'],
#    'norway': ['norway'],
#    'peru': ['peru'],
#    'philippines': ['philippines'],
#    'spain': ['spain'],
#    'sweden': ['sweden'],
#    'thailand': ['thailand'],
#    'turkey': ['turkey'],
#    'uruguay': ['uruguay'],
#    'venezuela': ['venezuela'], ## these is the original list
#}

import os 
cd = os.path.dirname(os.path.abspath(__file__))
ll_crisis_points=get_ll_crisis_points(os.path.join(cd ,'ll_crisis_dates.xlsx'),'import',list(country_dict.keys()))


#%%
if __name__ == '__main__':
    import ujson as json
    with open('crisis_points.json', 'w') as f:
        f.write(json.dumps(crisis_points))
    
