import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))
df = pd.read_csv("twitter_data.csv")
df.head()
df['labels'] = df['class'].map({0:"hate",1:"offensive",2:"none"})
df.head()
df = df[["tweet","labels"]]
df.head()


def clean(text):
    # Ensure the text is a string and convert to lowercase
    text = str(text).lower()

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Remove newlines
    text = re.sub(r'\n', '', text)

    # Remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


# Apply the clean function to the 'tweet' column of your dataframe
df["tweet"] = df["tweet"].apply(clean)
print(df.head())

x = np.array(df["tweet"])
y = np.array(df["labels"])

cv = CountVectorizer()
x = cv.fit_transform(x)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
from sklearn.metrics import accuracy_score
y_pred = clf.predict(X_test)
print(accuracy_score(y_test,y_pred,normalize=False))

data = "i would like to see someone blow your head off and kill you you fucking bitch"
df = cv.transform([data]).toarray()
print(clf.predict(df))

import pickle

with open('../../model/model_final.pkl','wb') as f:
    pickle.dump(clf,f)

with open('../../model/cv_final.pkl', 'wb') as c:
    pickle.dump(cv, c)