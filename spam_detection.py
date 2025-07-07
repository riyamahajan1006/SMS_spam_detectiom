import numpy as np
import pandas as pd

df = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')


# ---------------------------------------  Data preprocessing -------------------------------------------------------

df.info()
# drop extra columns
df.drop(columns=['Unnamed: 2',	'Unnamed: 3',	'Unnamed: 4'],inplace=True)
# rename columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
# encode target
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['target']=lb.fit_transform(df['target'])
#check for null values
df.isnull().sum()
# check for duplicates
df.duplicated().sum()
# if yes
df.drop_duplicates(keep='first',inplace=True)
df.duplicated().sum()


#---------------------------------------------- EDA ----------------------------------------------------------

df['target'].value_counts()
#pie chart
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="0.2f")
plt.show()
import nltk
nltk.download('punkt')
# no of characters in string
df['num_characters']=df['text'].apply(len)
# no of words in string
from nltk.tokenize import wordpunct_tokenize
df['text'] = df['text'].apply(lambda x: wordpunct_tokenize(x))
# no of sentences ina a string
import re
def simple_sent_tokenize(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

df['num_sentences'] = df['text'].apply(lambda x: len(simple_sent_tokenize(x)))
# describe
df[['num_characters','num_words','num_sentences']].describe()
df[df['target']==0][['num_characters','num_words','num_sentences']].describe()
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()
# histogram
import seaborn as sns
sns.histplot(df[df['target']==0]['num_characters'],color='pink')
sns.histplot(df[df['target']==1]['num_characters'])
#relation btw columns
sns.pairplot(df,hue='target')
  

# ------------------------------------------ text preprocessing -------------------------------------------------

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')
import string
string.punctuation
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
#function
def transform_text(text):
  text=text.lower()
  text=nltk.wordpunct_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return y
df['transformed_text']=df['text'].apply(transform_text)


# -------------------------------  word cloud for spam --------------------------------------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Make sure text is string
spam_wc = df[df['target']==1]['transformed_text'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

# 2. Join all text
text = spam_wc.str.cat(sep=" ")

# 3. Generate WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(text)

# 4. Plot
plt.figure(figsize=(4,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


# -------------------------------- word cloud for ham -----------------------------------------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Make sure text is string
spam_wc = df[df['target']==0]['transformed_text'].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))

# 2. Join all text
text = spam_wc.str.cat(sep=" ")

# 3. Generate WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(text)

# 4. Plot
plt.figure(figsize=(4,4))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()