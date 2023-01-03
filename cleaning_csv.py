import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
nltk.download('words')


df = pd.read_csv('twitter-news_1.csv', lineterminator='\n')
print(df.shape)
# print(df.head(10))
print(df.columns)

words = set(nltk.corpus.words.words())

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['twitter content'] = df['twitter content'].apply(lambda x: clean_text(x))

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  
                           u"\U0001F680-\U0001F6FF"   
                           u"\U0001F1E0-\U0001F1FF" 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

df['twitter content'] = df['twitter content'].apply(lambda x: remove_emoji(x))

lst = []
for i in range(df.shape[0]):
    lst.append(df['twitter content'][i])


list = []
for string in lst:
    new_string=re.sub('[^a-zA-Z0-9]',' ',string)
    cleaned_string=re.sub('\s+',' ',new_string)
    list.append(cleaned_string)

print(len(list))

new_df = pd.DataFrame(list, columns=['cleaned tweets'])

print('Shape of dataframe', new_df.shape)
print('Name of columns',new_df.columns)
print(new_df.info())

new_df.dropna(inplace=True)
# data[data.marks != 98]
new_df = new_df[new_df['cleaned tweets'] != ' newsejazah']

print('New shape of dataframe', new_df.shape)
new_df.to_csv('cleaned-twitter-news.csv', index=False)
# Check out the most used words in tweets related to it eda required in this task









