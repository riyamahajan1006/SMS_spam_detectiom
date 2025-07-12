import numpy as np
import pandas as pd

df = pd.read_csv(
    r'C:\Users\Riya Mahajan\OneDrive\Desktop\SMS_Spam_detection\spam.csv',
    encoding='latin1'
)

# --------------------------------------- Data loading & cleaning ---------------------------------------

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target labels
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['target'] = lb.fit_transform(df['target'])  # ham → 0, spam → 1

# Remove duplicates
# print(df.isnull().sum())
# print(df.duplicated().sum())
df.drop_duplicates(keep='first', inplace=True)

# ---------------------------------------------- EDA -----------------------------------------------------

import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

nltk.download('punkt')
from nltk.tokenize import word_tokenize

nltk.download('punkt')

df['num_words'] = df['text'].apply(lambda x: len(word_tokenize(x)))

# # Spam vs ham distribution
# plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%.2f")
# plt.title("Spam vs Ham distribution")
# plt.show()

# Add basic text statistics
df['num_characters'] = df['text'].apply(len)

import re
def simple_sent_tokenize(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

df['num_sentences'] = df['text'].apply(lambda x: len(simple_sent_tokenize(x)))

# # Visualize text length distributions
# plt.figure(figsize=(10,4))
# sns.histplot(df[df['target']==0]['num_characters'], color='blue', label='ham', kde=True)
# sns.histplot(df[df['target']==1]['num_characters'], color='red', label='spam', kde=True)
# plt.legend()
# plt.title("Message Length Distribution")
# plt.show()

# # Pairplot
# sns.pairplot(df, hue='target')

# ------------------------------------------ Text preprocessing ------------------------------------------

nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

df['transformed_text'] = df['text'].apply(transform_text)

# ---------------------------------------- Word Clouds ---------------------------------------------------

# # Word cloud for spam
# spam_wc_text = df[df['target']==1]['transformed_text'].str.cat(sep=" ")
# spam_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(spam_wc_text)

# plt.figure(figsize=(6,6))
# plt.imshow(spam_wc, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud - Spam")
# plt.show()

# # Word cloud for ham
# ham_wc_text = df[df['target']==0]['transformed_text'].str.cat(sep=" ")
# ham_wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white').generate(ham_wc_text)

# plt.figure(figsize=(6,6))
# plt.imshow(ham_wc, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud - Ham")
# plt.show()

# ------------------------------------- Most Frequent Words ---------------------------------------------

# # Spam
# spam_corpus = []
# for msg in df[df['target']==1]['transformed_text']:
#     spam_corpus.extend(msg.split())

# spam_freq = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['word', 'count'])
# plt.figure(figsize=(10,4))
# sns.barplot(x='word', y='count', data=spam_freq, palette='Reds_r')
# plt.xticks(rotation='vertical')
# plt.title("Top Words in Spam")
# plt.show()

# # Ham
# ham_corpus = []
# for msg in df[df['target']==0]['transformed_text']:
#     ham_corpus.extend(msg.split())

# ham_freq = pd.DataFrame(Counter(ham_corpus).most_common(30), columns=['word', 'count'])
# plt.figure(figsize=(10,4))
# sns.barplot(x='word', y='count', data=ham_freq, palette='Blues_r')
# plt.xticks(rotation='vertical')
# plt.title("Top Words in Ham")
# plt.show()

# ------------------------------------- Vectorisation ---------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer()  # optionally: max_features=3000

# Using TF-IDF because precision is important
X = tfidf.fit_transform(df['transformed_text']).toarray()
Y = df['target'].values

# ------------------------------------ Model Building ---------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# MultinomialNB (best for text classification)
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
y_pred = mnb.predict(X_test)

print("\n--- MultinomialNB Results ---")
print("Accuracy:", accuracy_score(Y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))

# # GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, Y_train)
# y_pred_gnb = gnb.predict(X_test)
# print("\n--- GaussianNB Results ---")
# print("Accuracy:", accuracy_score(Y_test, y_pred_gnb))
# print("Precision:", precision_score(Y_test, y_pred_gnb))

# # BernoulliNB
# bnb = BernoulliNB()
# bnb.fit(X_train, Y_train)
# y_pred_bnb = bnb.predict(X_test)
# print("\n--- BernoulliNB Results ---")
# print("Accuracy:", accuracy_score(Y_test, y_pred_bnb))
# print("Precision:", precision_score(Y_test, y_pred_bnb))

# -------------------------------- Notes & Suggestions --------------------------------------------------

# MultinomialNB with TF-IDF gave the best results here.
# You can experiment by:
# - Tuning vectorizer parameters
# - Trying other models (SVM, LogisticRegression, etc.)
# - Adding more features (like num_characters, num_words — although in tests they didn’t improve much)
# - Using MinMaxScaler did not help here.

import pickle
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))