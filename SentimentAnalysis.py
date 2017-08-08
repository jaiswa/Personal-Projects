
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

import os
## LOAD DATA AND SPLIT IT INTO TRAINING AND TESTING
rev_n = []
reveiw = []
path1 = './data/txt_sentoken/neg'
for f1 in os.listdir(path1):
    f1 = open(os.path.join(path1, f1),'r+')
    rev1 = f1.read()
    
    rev_n.append("".join(line.rstrip("\n") for line in rev1))

rev_p = []
path2 = './data/txt_sentoken/pos'
for f2 in os.listdir(path2):
    f2 = open(os.path.join(path2, f2),'r+')
    rev2 = f2.read()
   
    rev_p.append("".join(line.rstrip("\n") for line in rev2))
reveiw = rev_p + rev_n
print len(reveiw)
label1 = []
label2 = []
lab = 'neg'
for i in range(1000):
    label1.append(lab)
lab1 = 'pos'
for i in range(1000):
    label2.append(lab1)
label = []
label = label1 + label2

import pandas
messages = pandas.DataFrame(
    {'label': label,
     'message': reveiw,
    })
msg_train, msg_test, label_train, label_test =     train_test_split(messages['message'], messages['label'], test_size=0.05)

print len(msg_train), len(msg_test), len(msg_train) + len(msg_test)

rev_train = pandas.DataFrame(
    {'label': label_train,
     'message': msg_train,
    })

rev_test = pandas.DataFrame(
    {'label': label_test,
     'message': msg_test,
    })
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

#STEP2 DATA PREPROCESSING AND DATA TO VECTORS

bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(rev_train['message'])
train_bow = bow_transformer.transform(rev_train['message'])
tfidf_transformer_train = TfidfTransformer().fit(train_bow)

test_bow = bow_transformer.transform(rev_test['message'])
tfidf_test = tfidf_transformer_train.transform(test_bow)
tfidf_train = tfidf_transformer_train.transform(train_bow)
#get_ipython().magic(u"time reveiw_detector = MultinomialNB().fit(tfidf_train, rev_train['label'])")


# STEP3 TRAINING

%time reveiw_detector = MultinomialNB().fit(tfidf_train, rev_train['label']) 
# STEP4 TESTING AND FINDING SCORES 
alltest_predictions = reveiw_detector.predict(tfidf_test)
print 'accuracy', accuracy_score(rev_test['label'], alltest_predictions)
s = []
scores = reveiw_detector.predict_proba(tfidf_test)
for score in scores:
    s.append((-1)*score[0] + (1)*score[1])
#STEP4 SAVING THE RESULTS IN CSV FILE    
result = pandas.DataFrame(
    {'Reveiw_text': msg_test,
     'Actual_sentiment': label_test,
     'Computed_sentiment': alltest_predictions,
     'Sentiment_Score': s,
    })
result["Reveiw_text"] = result["Reveiw_text"].str[:50]
import csv
with open('Prob1_results.csv', 'wb') as csvfile:
    
    result.to_csv(csvfile, columns = ['Reveiw_text', 'Actual_sentiment', 'Computed_sentiment', 'Sentiment_Score'],
                             sep='\t',quoting=csv.QUOTE_NONE)

    


# In[ ]:



