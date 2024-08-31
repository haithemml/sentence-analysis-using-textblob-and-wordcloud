# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you
create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

output:
/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv
/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip
/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip

import matplotlib.pyplot as plt
import seaborn as sns
#load the dataset
sample_submission_data =pd.read_csv ("/kaggle/input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv")
print(sample_submission_data.head())

output:
   PhraseId  Sentiment
0    156061          2
1    156062          2
2    156063          2
3    156064          2
4    156065          2

#read tsv file
df = pd.read_csv('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip', sep='\t')
print(df.head())

output:
   PhraseId  SentenceId                                             Phrase  \
0         1           1  A series of escapades demonstrating the adage ...   
1         2           1  A series of escapades demonstrating the adage ...   
2         3           1                                           A series   
3         4           1                                                  A   
4         5           1                                             series   

   Sentiment  
0          1  
1          2  
2          2  
3          2  
4          2  

#the dataset contains four columns, we will handling two columns: Phrase column consists of textual reviews,and the Sentiment column contains corresponding numerical ratings.
#look at the column information
print(df.info())

output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 156060 entries, 0 to 156059
Data columns (total 4 columns):
 #   Column      Non-Null Count   Dtype 
---  ------      --------------   ----- 
 0   PhraseId    156060 non-null  int64 
 1   SentenceId  156060 non-null  int64 
 2   Phrase      156060 non-null  object
 3   Sentiment   156060 non-null  int64 
dtypes: int64(3), object(1)
memory usage: 4.8+ MB
None

EXPLORARY DATA/

Now let's look at this data step by step. We'll start by analyzing the distribution of reviews, which will give us insight into overall PhraseId sentiment. Subsequently, we will be able to deepen our analysis, in particular by studying the length of the reviews and possibly drawing lessons from the textual content of the latter.
we’ll analyze the length of the phrases, as this can sometimes correlate with the sentiment of feedback. We will first calculate the length of each phrase and then visualize the data:

# Calculating the length of each phrase
df['Phrase Length'] = df['Phrase'].apply(len)

# Plotting the distribution of review lengths
plt.figure(figsize=(9, 6))
sns.histplot(df['Phrase Length'], bins=50, kde=True)
plt.title('Distribution of Phrases Lengths')
plt.xlabel('Length of Phrase')
plt.ylabel('Count')
plt.show()

output:
/opt/conda/lib/python3.10/site-packages/seaborn/_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version.
Convert inf values to NaN before operating instead.
  with pd.option_context('mode.use_inf_as_na', True):
  
from textblob import TextBlob

def textblob_sentiment_analysis(Phrase):
    # Analyzing the sentiment of the review
    sentiment = TextBlob(Phrase).sentiment
    # Classifying based on polarity
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Applying TextBlob sentiment analysis to the review
df['Sentiment_sit'] = df['Phrase'].apply(textblob_sentiment_analysis)

print(df.head())

output:
   PhraseId  SentenceId                                             Phrase  \
0         1           1  A series of escapades demonstrating the adage ...   
1         2           1  A series of escapades demonstrating the adage ...   
2         3           1                                           A series   
3         4           1                                                  A   
4         5           1                                             series   

   Sentiment  Phrase Length Sentiment_sit  
0          1            188      Positive  
1          2             77      Positive  
2          2              8       Neutral  
3          2              1       Neutral  
4          2              6       Neutral  

#we visualize the distribution of feelings.

Now, The dataset includes sentiment labels for each phrase, classified as Positive, Negative, or Neutral based on the polarity score calculated by TextBlob.
import matplotlib.pyplot as plt

#Count the occurrences of each feeling
sentiment_counts = df['Sentiment_sit'].value_counts()

#Visualize the distribution of feelings
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Phrases')
plt.show()

output:

#Analyze sentence length by sentiment To explore whether sentence length impacts sentiment, you can create visualizations or statistics.
# Calculate the average sentence length for each sentiment
mean_phrase_length = df.groupby('Sentiment_sit')['Phrase Length'].mean()

# View average sentence length by sentiment
mean_phrase_length.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Average Phrase Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Average Phrase Length')
plt.show()

output:

#Sentiment distribution by sentence length We can also view the distribution of sentence lengths for each sentiment:
import seaborn as sns

# Visualize the distribution of sentence lengths for each feeling
sns.boxplot(x='Sentiment_sit', y='Phrase Length', data=df, palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'})
plt.title('Phrase Length Distribution by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Phrase Length')
plt.show()

output:

#Exploring extreme sentences We may want to explore the most positive or negative sentences
# Explore the most positive sentences
print(df[df['Sentiment_sit'] == 'Positive'].sort_values(by='Phrase Length', ascending=False).head())

# Explore the most negative sentences
print(df[df['Sentiment_sit'] == 'Negative'].sort_values(by='Phrase Length', ascending=False).head())

output:
        PhraseId  SentenceId  \
105155    105156        5555   
22534      22535        1020   
9650        9651         403   
11800      11801         509   
142011    142012        7705   

                                                   Phrase  Sentiment  \
105155  ... spiced with humor -LRB- ' I speak fluent f...          3   
22534   For every cheesy scene , though , there is a r...          3   
9650    Notwithstanding my problem with the movie 's f...          3   
11800   I stopped thinking about how good it all was ,...          4   
142011  Sitting in the third row of the IMAX cinema at...          3   

        Phrase Length Sentiment_sit  
105155            283      Positive  
22534             263      Positive  
9650              261      Positive  
11800             260      Positive  
142011            259      Positive  
       PhraseId  SentenceId  \
43802     43803        2124   
85187     85188        4406   
43805     43806        2124   
82375     82376        4255   
43806     43807        2124   

                                                  Phrase  Sentiment  \
43802  -LRB- City -RRB- reminds us how realistically ...          3   
85187  The film was produced by Jerry Bruckheimer and...          1   
43805  reminds us how realistically nuanced a Robert ...          2   
82375  Even if the enticing prospect of a lot of nubi...          0   
43806  reminds us how realistically nuanced a Robert ...          3   

       Phrase Length Sentiment_sit  
43802            279      Negative  
85187            266      Negative  
43805            262      Negative  
82375            260      Negative  
43806            260      Negative  

#The sentiment analysis performed on the dataset reveals two distinct groups of sentences. Here is a summary of the findings based on the observed results:

Phrases with Positive Feeling:

Sentences identified with positive sentiment were relatively long, with an average of over 260 characters. These sentences come from various parts of the dataset, indicating that they are associated with positive reviews or comments. Positive feelings are mostly associated with sentences that, although long, contain appreciated nuances and complexities, as demonstrated by the vocabulary used in these sentences. Sentences with Negative Feeling:

Phrases identified with negative sentiment are similar in length to those with positive sentiment, around 260 to 279 characters. These phrases express criticism or negative opinions, often focused on specific aspects of the experiences or products mentioned. The vocabulary used in these sentences is often critical, which explains their classification as "negative". General Observations:

Sentence length seems to be an important indicator in the expression of feelings, but it is not the only determining factor. Sentence content and nuance play a key role in sentiment analysis. Longer sentences tend to express stronger feelings, whether positive or negative, suggesting deeper thought or stronger emotional engagement in those expressions.


#examining the most frequently occurring words in positive, negative, and neutral reviews using a word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming df is your DataFrame containing the sentiment analysis
# df['Sentiment_sit'] should contain 'Positive', 'Negative', or 'Neutral'

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Filter reviews by sentiment and combine them into one large string
positive_reviews = " ".join(df[df['Sentiment_sit'] == 'Positive']['Phrase'].tolist())
negative_reviews = " ".join(df[df['Sentiment_sit'] == 'Negative']['Phrase'].tolist())
neutral_reviews = " ".join(df[df['Sentiment_sit'] == 'Neutral']['Phrase'].tolist())

# Generate and display word clouds
generate_wordcloud(positive_reviews, 'Positive Reviews Word Cloud')
generate_wordcloud(negative_reviews, 'Negative Reviews Word Cloud')
generate_wordcloud(neutral_reviews, 'Neutral Reviews Word Cloud')

output:

#Interpreting the Word Clouds Positive Reviews Word Cloud: This will show the most frequent words used in reviews classified as positive. Common positive terms may be emphasized, giving you insight into what aspects of the product or service are most appreciated. Negative Reviews Word Cloud: This cloud will highlight the words most often used in negative reviews, which can help identify common pain points or areas needing improvement. Neutral Reviews Word Cloud: The neutral word cloud will focus on words that are commonly used in reviews that don't express a strong sentiment either way.

#We can customize the appearance of the word clouds by modifying the WordCloud parameters, such as changing the max_words, colormap, or excluding stopwords.
from wordcloud import STOPWORDS

# Using default stopwords from the wordcloud library
stopwords_list = set(STOPWORDS)

# Optionally, add your own custom stopwords
custom_stopwords = {"movie", "film", "one"}  # Example words to exclude
stopwords_list.update(custom_stopwords)
# Assuming df is your DataFrame and 'Phrase' contains the text data
positive_reviews_text = " ".join(df[df['Sentiment_sit'] == 'Positive']['Phrase'].tolist())
negative_reviews_text = " ".join(df[df['Sentiment_sit'] == 'Negative']['Phrase'].tolist())
neutral_reviews_text = " ".join(df[df['Sentiment_sit'] == 'Neutral']['Phrase'].tolist())

# Generate and display word clouds
def generate_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        max_words=100, 
        colormap='viridis',
        stopwords=stopwords_list
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title)
    plt.show()



