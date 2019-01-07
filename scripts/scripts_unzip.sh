cd Financial_Times
for i in *.zip;
do unzip -q "$i" -d "${i%%.zip}";
done
rm *.zip


