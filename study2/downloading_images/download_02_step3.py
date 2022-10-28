import pandas as pd
import numpy as np
import urllib.request 
import os
import glob
import os.path
import csv
import re
import pathlib
import mimetypes
from urllib.request import urlopen
from pathlib import Path


df = pd.read_csv('/project/ll_774_951/uk_ru/twitter/step2/csv_2022_02_url.csv')
# dff = df.values.tolist()
counter = 806
for link in df['urls'][1030:30000]:
    path = f"/project/ll_774_951/uk_ru/twitter/step3/{counter}"+Path(link).suffix
    try:
        urllib.request.urlretrieve(link,path)
        counter += 1
    except:
        pass