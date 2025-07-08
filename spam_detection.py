import numpy as np
import pandas as pd

# Read the dataset
df = pd.read_csv(
    r'C:\Users\Riya Mahajan\OneDrive\Desktop\SMS_Spam_detection\spam.csv',
    encoding='latin1'
)


# ---------------------------------------  Data preprocessing -------------------------------------------------------

# Drop extra columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['target'] = lb.fit_transform(df['target'])

# Check for and remove duplicates
# print(df.isnull().sum())
# print(df.duplicated().sum())
df.drop_duplicates(keep='first', inplace=True)

# ---------------------------------------------- EDA ----------------------------------------------------------

# import matplotlib.pyplot as plt
# plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%.2f")
# plt.show()

import nltk
nltk.download('punkt')

# Number of characters in each message
df['num_characters'] = df['text'].apply(len)

# Number of words
from nltk.tokenize import wordpunct_tokenize
df['num_words'] = df['text'].apply(lambda x: len(wordpunct_tokenize(x)))

# Number of sentences
import re
def simple_sent_tokenize(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

df['num_sentences'] = df['text'].apply(lambda x: len(simple_sent_tokenize(x)))

# Uncomment below to see histograms or pairplots
# import seaborn as sns
# sns.histplot(df[df['target']==0]['num_characters'], color='pink')
# sns.histplot(df[df['target']==1]['num_characters'])
# sns.pairplot(df, hue='target')

# ------------------------------------------ Text preprocessing -------------------------------------------------

nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    words = nltk.wordpunct_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

df['transformed_text'] = df['text'].apply(transform_text)

# -------------------------------------Vectorisation----------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

# Using TF-IDF because precision matters
X = tfidf.fit_transform(df['transformed_text']).toarray()
Y = df['target'].values

# ------------------------------------Model building ----------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

mnb = MultinomialNB()
mnb.fit(X_train, Y_train)

y_pred = mnb.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))

# ------------------------------Notes------------------------------------------------------------
# GaussianNB & BernoulliNB gave worse results than MultinomialNB.
# TF-IDF with MultinomialNB worked best here.
# Other improvements possible: tune vectorizer params, try other models.
