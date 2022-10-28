import pandas as pd
import numpy as np
import urllib.request 
import os
import glob
import os.path
import csv
import re

new_file_path=[]
path_2022_02 = "/project/ll_774_951/uk_ru/twitter/step1/*.csv"
for fname in glob.glob(path_2022_02):
    new_file_path.append(fname)

new_list = []
special_char = ",]['"
for csv_path in new_file_path:
    tempdf = pd.read_csv(csv_path,engine='python',error_bad_lines=False)
    new_list.append(tempdf)
final_df = pd.concat(new_list)


def find_unique(final_df):
    df_all_urls = final_df.melt(value_name='urls')[['urls']] # into single column
    download_1 = df_all_urls.loc[df_all_urls.urls.str.contains('http', na=False)]
    download_1 = download_1.replace("\[\'",'',regex=True)
    download_1 = download_1.replace("\'\]",'',regex=True)
    download_1 = download_1.replace("\,",'',regex=True)
    download_1 = download_1.drop_duplicates()
    return download_1 # a one column dataframe

one_col_uni_url = find_unique(final_df)
one_col_uni_url.to_csv('/project/ll_774_951/uk_ru/twitter/step2/csv_2022_02_url.csv',index=False)