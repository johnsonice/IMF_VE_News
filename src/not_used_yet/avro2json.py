"""
Convert avro files to json
"""
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter
import avro.schema
import ujson as json
import os
from glob import glob
import argparse


#Defining DJDNA Snapshots Schema May 22,2018
djdna_avro_schema = {
"type":"record",
"name":"Delivery",
"namespace":"com.dowjones.dna.avro",
"doc":"Avro schema for extraction content used by Dow Jones' SyndicationHub",
"fields":[
    {"name":"an","type":["string","null"]},
    {"name":"modification_datetime","type":["long","null"]},
    {"name":"ingestion_datetime","type":["long","null"]},
    {"name":"publication_date","type":["long","null"]},
    {"name":"publication_datetime","type":["long","null"]},
    {"name":"snippet","type":["string","null"]},
    {"name":"body","type":["string","null"]},
    {"name":"art","type":["string","null"]},
    {"name":"action","type":["string","null"]},
    {"name":"credit","type":["string","null"]},
    {"name":"byline","type":["string","null"]},
    {"name":"document_type","type":["string","null"]},
    {"name":"language_code","type":["string","null"]},
    {"name":"title","type":["string","null"]},
    {"name":"copyright","type":["string","null"]},
    {"name":"dateline","type":["string","null"]},
    {"name":"source_code","type":["string","null"]},
    {"name":"modification_date","type":["long","null"]},
    {"name":"section","type":["string","null"]},
    {"name":"company_codes","type":["string","null"]},
    {"name":"publisher_name","type":["string","null"]},
    {"name":"region_of_origin","type":["string","null"]},
    {"name":"word_count","type":["int","null"]},
    {"name":"subject_codes","type":["string","null"]},
    {"name":"region_codes","type":["string","null"]},
    {"name":"industry_codes","type":["string","null"]},
    {"name":"person_codes","type":["string","null"]},
    {"name":"currency_codes","type":["string","null"]},
    {"name":"market_index_codes","type":["string","null"]},
    {"name":"company_codes_about","type":["string","null"]},
    {"name":"company_codes_association","type":["string","null"]},
    {"name":"company_codes_lineage","type":["string","null"]},
    {"name":"company_codes_occur","type":["string","null"]},
    {"name":"company_codes_relevance","type":["string","null"]},
    {"name":"source_name","type":["string","null"]}
]
}
read_schema = avro.schema.Parse(json.dumps(djdna_avro_schema))

# open avro and extract data per schema 
def avro2json(file):
    with DataFileReader(open(file, "rb"), DatumReader(read_schema)) as reader:
        docs = []
        for doc in reader:
            docs.append(doc)
        reader.close()
    docs = [json.loads(json.dumps(d)) for d in docs]
    return docs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', action='store', dest='in_dir', required=True)
    parser.add_argument('-o', '--out_dir', action='store', dest='out_dir', required=True)
    parser.add_argument('-v', '--verbose', action='store', dest='verbose', default=False)
    args = parser.parse_args()
    
    av_docs = glob(args.in_dir + "\*.avro")
    for doc in av_docs:
        if args.verbose:
            print('extracting file {}...'.format(os.path.basename(doc)))
        json_docs = avro2json(doc)
        if args.verbose:
            print("writing extracted docs to json...\n\n")
        for d in json_docs:
            f_name = os.path.join(args.out_dir, d['an'] + ".json")
            with open(f_name, 'w') as f:
                f.write(json.dumps(d))
