import pandas as pd
import numpy as np
import urllib.request 
import os
import glob
import os.path

file_paths=[]
path_2022_02 = "/project/ll_774_951/uk_ru/twitter/data/2022-02/*.csv"
for fname in glob.glob(path_2022_02):
    file_paths.append(fname)
path_2022_03 = "/project/ll_774_951/uk_ru/twitter/data/2022-03/*.csv"
for fname in glob.glob(path_2022_03):
    file_paths.append(fname)
path_2022_04 = "/project/ll_774_951/uk_ru/twitter/data/2022-04/*.csv"
for fname in glob.glob(path_2022_04):
    file_paths.append(fname)
file_paths.sort()

for file in file_paths:
    df = pd.read_csv(file,  engine='python', error_bad_lines=False)
    
    tweet_id = df['tweet_id']
    
    
    for i in tweet_id:
      url = "https://twitter.com/anyuser/status/" + i
