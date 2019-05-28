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
