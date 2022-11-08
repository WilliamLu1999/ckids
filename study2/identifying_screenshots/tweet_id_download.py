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
import tweetcapture

df = pd.read_csv('/Users/William/Downloads/tweet_id_url.csv') # 记得改
# dff = df.values.tolist()
#counter = 4001
for link in df['0'][4000::]:
    #path = f"/Users/William/Downloads/tweet_id_pic/{counter}"+Path(link).suffix
    try:
        os.system('tweetcapture '+str(link))
        #urllib.request.urlretrieve(link,path)
        #counter += 1
    except:
        pass
print('success')