import numpy as np
import os
import shutil

# sample source and destination
source = '/data/News_data_raw/Financial_Times_processed'
dest = '/data/News_data_raw/tiny_test/Financial_Times_processed'

# list all files in dir
files = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

# sample 0.01 of the files randomly
random_files = np.random.choice(files, int(len(files)*.01))

# copy over the sampled files
for file in random_files:
    old_path = os.path.join(source, file)
    shutil.copy(old_path, dest)

print("Done. Coppied {} files from {} to {}".format(len(random_files), source, dest))

