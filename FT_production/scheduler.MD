## simple tutorial 


crontab -e

## example: run at 23:25 p.m every day with:
# min hour dayofmonth month dayofweek
# 25 23 * * * bash [somefile.sh]

```
25 23 * * * bash /home/chuang/Dev/IMF_VE_news/FT_production/update.sh >> /data/News_data_raw/Production/data/logs/run_evening.log
0 7 * * * bash /home/chuang/Dev/IMF_VE_news/FT_production/update.sh >> /data/News_data_raw/Production/data/logs/run_morning.log

```

### usually if you need to use conda environment you need to set you conda path
### in your bash file 
```
#HOME=/root
#LOGNAME=root
#PATH=/usr/local/sbin:/user/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
PATH=/home/chuang/anaconda3/bin:/home/chuang/perl5/bin:/usr/local/bin:/usr/bin
LANG=en_US.UTF-8
SHELL=/bin/sh
#PWD=/root

now=$(date +"%T")
echo "Current time : $now" 
cd /home/chuang/Dev/IMF_VE_news/FT_production/
source activate nlp
python FT_update.py
now=$(date +"%T")
echo "Finished : $now" 

```

