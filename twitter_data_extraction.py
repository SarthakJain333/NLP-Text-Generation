import pandas as pd
from tqdm import tqdm
import snscrape.modules.twitter as sntwitter

scraper = sntwitter.TwitterSearchScraper("News")
print(scraper)
x = []

for i,tweet in tqdm(enumerate(scraper.get_items()), total=10000):
    data = [tweet.id, tweet.content]
    x.append(data)
    if i > 10000:
        break

df = pd.DataFrame(x)
print(df.head(10))

df.to_csv('tweets_news.csv')

