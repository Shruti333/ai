# -*- coding: utf-8 -*-
"""Untitled28.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tE7Mc-Ow7DxBq0VWL_BRUZsoEXmOZXPU
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import re
import sys
import nltk
nltk.download('stopwords')

import matplotlib.pyplot as plt
# %matplotlib inline
data_source_url = "/content/Tweets.csv.zip"
airline_tweets = pd.read_csv(data_source_url)
airline_tweets.head()
plot_size = plt.rcParams["figure.figsize"]

print(plot_size[0])

print(plot_size[1])
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size


airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values
processed_features = []
for sentence in range(0, len(features)):


     processed_feature = re.sub(r'\\W', ' ', str(features[sentence]))

     processed_feature= re.sub(r'\\s+[a-zA-Z]\\s+', ' ', processed_feature)

     processed_feature = re.sub(r'\\^[a-zA-Z]\\s+', ' ', processed_feature)

     processed_feature = re.sub(r'\\s+', ' ', processed_feature, flags=re.I)

     processed_feature = re.sub(r'^b\\s+', '', processed_feature)

     processed_feature = processed_feature.lower()

     processed_features.append(processed_feature)



from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred ))


 #Bigrams
bigrm = list(nltk.bigrams('stopwords'))
words_2 = nltk.FreqDist(bigrm)
words_2.plot(20, color='salmon', title='Bigram Frequency')

#Trigrams
trigrm = list(nltk.trigrams('stopwords'))
words_2 = nltk.FreqDist(trigrm)
words_2.plot(20, color='salmon', title='trigram Frequency')