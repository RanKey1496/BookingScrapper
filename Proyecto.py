# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 20:22:08 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
import requests
import os
import numpy as np
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
def classification(res, senticon, x1, x2):
    polarity = 1
    sumWeight = 0
    for r in res:
        for word in r:
            if (is_negative(word['token'])):
                polarity = -1
            if (word['tag'][0] == 'A'):
                e = exists(senticon, word['lemma'])
                if (e):
                    sumWeight += float(e[1])*polarity
            if (polarity == -1 and word['token'] == '.'):
                polarity = 1
    if (sumWeight > 0):
        x1 += 1
    else:
        x2 += 1
    return x1, x2
                        
# In[6]:
def obtain_freeling(fileName):
    files = {'file': open(fileName, 'r')}
    params = {'outf': 'tagged', 'format': 'json'}
    url = 'http://www.corpus.unam.mx/servicio-freeling/analyze.php'
    req = requests.post(url, files=files, params=params)
    return req.json()

# In[6]:
def get_database():
    db = open('reviews.txt', 'r')
    db_data = []
    db_target = []
    for line in db:
        db_data.append(line.split('\t')[0])
        db_target.append(line.split('\t')[1][:-1])
    db.close()
    return db_data, db_target

# In[7]:
def model_classification(data, y, model):
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
        acc.append(accuracy_score(Xtest, Ytest))
    return acc, sens, spec

# In[7]:
if __name__ == '__main__':
    write_comments_in_separeted_files()
    senticon = get_senticon()
    data, target = get_database()
    x1, x2 = 0, 0
    files = get_file_names()
    for i in files:
        review = './reviews/' + i
        req = obtain_freeling(review)
        res1, res2 = classification(req, senticon)
        if ()
        x1 += res1; x2 += res2;
    print(x1, x2)
    # print_results(model_classification(data, target, LR()), 'Logistic Regression')
    # print_results(model_classification(data, target, KNN()), 'K-Nearest Neighbors')
    # print_results(model_classification(data, target, RF()), 'Random Forest')
    
# In[8]:
    length = len(data)
    print('Porcentaje de buenos comentarios: ', (x1/length)*100)
    print('Porcentaje de malos comentarios: ', (x2/length)*100)