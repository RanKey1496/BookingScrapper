# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 20:22:08 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
import requests
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF

# In[2]:
def write_comments_in_separeted_files():
    file = open('reviews.txt', 'r').readlines()
    for i, line in enumerate(file):
        newFile = open('reviews/review'+str(i)+'.txt', 'w')
        newFile.write(file[i].split('\t')[0])
        newFile.close()
        
def get_file_names():
    for root, dirs, files in os.walk('./reviews'):
        return files

def print_results(acc, sens, esp, model):
    print('\nResultados con ' + model + '\n')
    print('Accuray: ', np.mean(acc), '+/-', np.std(acc))
    print('Sensitivity: ', np.mean(sens), '+/-', np.std(sens))
    print('Specificity: ', np.mean(esp), '+/-', np.std(esp))
    
def error_measures(Ypredict, Yreal):
    CM = confusion_matrix(Yreal, Ypredict)
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    return sens, spec

def is_negative(text):
    low = text.lower()
    negatives = ['no', 'pero', 'aunque', 'but', 'ni']
    if low in negatives:
        return True
    return False

# In[3]:
def get_senticon():
    listSenticon = []
    mlSenticon = open('MLSenticon.txt', 'r')
    for i in mlSenticon:
        word = i.split('\t')
        listSenticon.append([word[0], word[1].replace('\n', '')])
    mlSenticon.close()
    return listSenticon
    
# In[4]:
def exists(array, word):
    for i in array:
        if (i[0] == word):
            return i
    return None

# In[5]:
def classification(res, senticon):
    x1, x2 = 0, 0
    polarity = 1
    for r in res:
        for word in r:
            if (is_negative(word['token'])):
                polarity = -1
            if (word['tag'][0] == 'A'):
                e = exists(senticon, word['lemma'])
                if (e):
                    if (float(e[1])*polarity > 0):
                        x1 += float(e[1])
                    else:
                        x2 += float(e[1])
            if (polarity == -1 and word['token'] == '.'):
                polarity = 1
    return x1, x2
                        
# In[6]:
def obtain_freeling(fileName):
    files = {'file': open(fileName, 'r')}
    params = {'outf': 'tagged', 'format': 'json'}
    url = 'http://www.corpus.unam.mx/servicio-freeling/analyze.php'
    req = requests.post(url, files=files, params=params)
    return req.json()

def get_database():
    db = open('reviews.txt', 'r')
    db_data = []
    db_target = []
    for line in db:
        db_data.append(line.split('\t')[0])
        db_target.append(line.split('\t')[1][:-1])
    db.close()
    return db_data, db_target

# In[6]:
def lr_results(data, target):
    lr = lr_classification(data, target)
    data = {'Accurancy': np.mean(lr[0]), 'Sensibility': np.mean(lr[1]), 'Specificity': np.mean(lr[2])}
    df = pd.DataFrame(data, index=[1])
    print('\nLogistic Regression Classification')
    print(df)

def knn_results(data, target):
    ks = [1,3,5,7,9,15,25]
    acc = []
    sens = []
    spec = []
    for i in ks:
        knn = knn_classification(data, target, i)
        acc.append(np.mean(knn[0]))
        sens.append(np.mean(knn[1]))
        spec.append(np.mean(knn[2]))
    data = {'Accurancy': acc, 'Sensibility': sens, 'Specificity': spec}
    df = pd.DataFrame(data, index=ks)
    print('\nK-Near Neighbors Classification')
    print(df)
    
def rf_results(data, target):
    n_estimators = [10,20,30,40,50]
    acc = []
    sens = []
    spec = []
    for i in n_estimators:
        rf = rf_classification(data, target, i)
        acc.append(np.mean(rf[0]))
        sens.append(np.mean(rf[1]))
        spec.append(np.mean(rf[2]))
    data = {'Accurancy': acc, 'Sensibility': sens, 'Specificity': spec}
    df = pd.DataFrame(data, index=n_estimators)
    print('\nRandom Forest Classification')
    print(df)

# In[7]:
def lr_classification(data, y):
    model = LR()
    acc = []
    sens = []
    spec = []
    
    for i in range(100):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, y)
        model.fit(Xtrain, Ytrain)
        y_pred = model.predict(Xtest)
        sen, spc = error_measures(y_pred, Ytest)
        sens.append(sen)
        spec.append(spc)
        acc.append(model.score(Xtest, Ytest))
    return acc, sens, spec

def knn_classification(data, y, k):
    model = KNN(n_neighbors=k)
    acc = []
    sens = []
    spec = []
    
    for i in range(100):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, y)
        model.fit(Xtrain, Ytrain)
        y_pred = model.predict(Xtest)
        sen, spc = error_measures(y_pred, Ytest)
        sens.append(sen)
        spec.append(spc)
        acc.append(model.score(Xtest, Ytest))
    return acc, sens, spec

def rf_classification(data, y, estimators):
    model = RF(n_estimators=estimators)
    acc = []
    sens = []
    spec = []
    
    for i in range(100):
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, y)
        model.fit(Xtrain, Ytrain)
        y_pred = model.predict(Xtest)
        sen, spc = error_measures(y_pred, Ytest)
        sens.append(sen)
        spec.append(spc)
        acc.append(accuracy_score(Ytest, y_pred))
    return acc, sens, spec

# In[7]:
if __name__ == '__main__':
    write_comments_in_separeted_files()
    senticon = get_senticon()
    data, target = get_database()
    x = []
    files = get_file_names()
    for i in files:
        review = './reviews/' + i
        req = obtain_freeling(review)
        result = classification(req, senticon)
        x.append([result[0], result[1]])
    print(x)
    
# In[8]:

    
# In[9]:
    # Evaluando con x1, x2
    lr_results(x, target)
    knn_results(x, target)
    rf_results(x, target)
    
# In[10]:
    data, target = get_database()
    vectorTF = TfidfVectorizer()
    vectorTF.fit(data)
    bow = vectorTF.transform(data)
    lr_results(bow, target)
    knn_results(bow, target)
    rf_results(bow, target)
    
# In[11]:
    data, target = get_database()
    vectorCount = CountVectorizer(ngram_range=(1,2))
    vectorCount.fit(data)
    bow = vectorCount.transform(data)
    lr_results(bow, target)
    knn_results(bow, target)
    rf_results(bow, target)