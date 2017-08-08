

import numpy as np
import math
from textblob import TextBlob
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import sys
f = open('prob1_input.txt', 'r') #set the path accordingly
data = f.readlines()
int1 = eval(data[0])

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]
def termFrequency(term, document):
    normalizeDocument = document.lower().split()
    return normalizeDocument.count(term.lower()) / float(len(normalizeDocument))
    #normalized TermFrequency calculation.

def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in range(len(allDocuments)):
        if term.lower() in allDocuments[doc].lower().split():
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1
 
    if numDocumentsWithThisTerm > 0:
        return 1.0 + math.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0
    
q_sum = 0   
setA = []
setB = []

i=0
for index in range(1,int1+1):  
    setA.append(data[index])
    
for index in range(int1+2,len(data)):  
    setB.append(data[index])
b = len(setB)

sum_d = np.zeros((b))
sum_dq = np.zeros((b))
cos_smlrty = np.zeros((b))

for index1 in range(len(setA)):
    words=split_into_lemmas(setA[index1])
    
    a = len(words)
    idf = np.ones((a))
    tf_array = np.ones((a, b))
    tf_q = np.ones((a))
    tf_qd = np.ones((a))
    idf_q = np.ones((a))
    #tf_idf calculation     
    for i,word in enumerate(words):
        idf[i] = inverseDocumentFrequency(str(word), setB)
        tf_q[i]=(termFrequency(str(word),setA[index1]))
        
    for i in range (len(tf_q)):
        q_sum = q_sum + tf_q[i]*tf_q[i] 
        q_sum = math.sqrt(q_sum)
    for index2 in range(len(setB)):
        for i,word in enumerate(words):
            tf_array[i][index2] = (termFrequency(str(word),setB[index2]))
            tf_array[i,index2]=tf_array[i,index2]*idf[i]
        temp = tf_array[:,index2]
     #cos similarity calculation   
        for i in range(len(temp)):
            sum_d[index2] = sum_d[index2] + temp[i]*temp[i]
            sum_dq[index2] = tf_q[i]*temp[i] + sum_dq[index2]   
           
        sum_d[index2] = math.sqrt(sum_d[index2])
        cos_smlrty[index2] = sum_dq[index2]/(sum_d[index2]*q_sum)
    print (1+np.argmax(cos_smlrty))
    
            
        




