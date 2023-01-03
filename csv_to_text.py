import pandas as pd
import numpy as np

df = pd.read_csv('cleaned-twitter-news.csv')

a = list(df['cleaned tweets'])

with open('Twitter-news.txt', 'w') as f:
    for item in a:
        f.write("%s\n" % item)
    print('Done')


