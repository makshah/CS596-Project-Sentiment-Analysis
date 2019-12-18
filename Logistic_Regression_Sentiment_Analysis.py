import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import string
import nltk
import time
import warnings 
import sklearn.metrics as acc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# PRE PROCESSING THE CSV FILES TO REMOVE THE UNWANTED CHARACTERS/WORDS TO CLEAN THE DATA

# Reading both the CSV files

training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')

# combining both the files to one single file

combined_csv = training_data.append(testing_data, ignore_index=True)

def pattern_edittor(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

combined_csv['clean_tweet']= np.vectorize(pattern_edittor)(combined_csv['tweet'],"@[\w]*")
combined_csv['clean_tweet']= combined_csv['clean_tweet'].str.replace("[^a-zA-Z#]"," ")
combined_csv['clean_tweet']= combined_csv['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

tokenized_tweet = combined_csv['clean_tweet'].apply(lambda x: x.split())
#tokenized_tweet.head(10)

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
#tokenized_tweet.head(10)

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combined_csv['clean_tweet'] = tokenized_tweet
print("Using Bag of Words method:")
bag_of_words_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bag_of_words = bag_of_words_vectorizer.fit_transform(combined_csv['clean_tweet'])

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combined_csv['clean_tweet'])

train_bag_of_words = bag_of_words[:31962,:]
test_bag_of_words = bag_of_words[31962:,:]

# Data i split into train and test
xtrain_bag_of_words, xvalid_bag_of_words, ytrain, yvalid = train_test_split(train_bag_of_words, training_data['label'], random_state=42, test_size=0.3)
#Using the Logistic Regression method
lreg = LogisticRegression()
lreg.fit(xtrain_bag_of_words, ytrain) # training the model
t0 = time.time()
prediction = lreg.predict_proba(xvalid_bag_of_words) # predicting on the validation set
threshold = 0.5
prediction_int = prediction[:,1] >= threshold 
prediction_int = prediction_int.astype(np.int)

# Calculating F1 Score:

score_bag_of_words = f1_score(yvalid, prediction_int)
accuracy_bag_of_words = acc.accuracy_score(yvalid, prediction_int)
print('Logistic Regression F1 score using Bag of Words method is:', score_bag_of_words)
print('Logistic Regression Accuraccy using Bag of Words method is:', accuracy_bag_of_words,'\nWhich is equivalent to:',("%.2f" % (accuracy_bag_of_words*100)),'%')
t1 = time.time()
test_pred = lreg.predict_proba(test_bag_of_words)
test_pred_int = test_pred[:,1] >= threshold
test_pred_int = test_pred_int.astype(np.int)
time_train = t0
time_predict = t1 - t0
print("Bag of Words Training time: %fs" % (time_train))
print("Bag of Words Prediction time: %fs" % (time_predict))
testing_data['label'] = test_pred_int
submission = testing_data[['tweet','label']]

# Storing data in CSV file

submission.to_csv('Logistic_bag_of_words.csv', index=False) 

# Similarly for TF-IDF Method
print("\n\nUsing TF-TDF method:")
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)
t0 = time.time()
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= threshold
prediction_int = prediction_int.astype(np.int)

score_tfdif = f1_score(yvalid, prediction_int)
accuracy_tfidf = acc.accuracy_score(yvalid, prediction_int)
print('Logistic Regression F1 Score using TF-IDF method is:', score_tfdif)
print('Logistic Regression prediction using TF-IDF method is:', accuracy_tfidf ,'\nWhich is equivalent to:',"%.2f" % (accuracy_tfidf*100),'%')
t1 = time.time()
test_pred = lreg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= threshold
test_pred_int = test_pred_int.astype(np.int)
time_train = t0
time_predict = t1 - t0
print("TF-IDF Training time: %fs and Prediction time: %fs" % (time_train, time_predict))
testing_data['label'] = test_pred_int
submission = testing_data[['tweet','label']]

# Storing data in CSV File
submission.to_csv('Logistic_TFIDF.csv', index=False) 