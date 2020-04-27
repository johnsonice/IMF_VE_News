import math
month_of_event = 5
year_of_event = 2018

# Mutable window for discovery
months_prior = 18

# includes partial first month and the entire month of event if True
partial_months_as_whole = True

# Months to read data from
if partial_months_as_whole:
    months_prior += 1
start_year = year_of_event - int(math.ceil((months_prior - month_of_event)/12))
start_month = 12 - (months_prior-month_of_event)%12 + 1
start_day = 8

print("Start y/m/d: "+str(start_year)+"/"+str(start_month)+"/"+str(start_day)+"\n")