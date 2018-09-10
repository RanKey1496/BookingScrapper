# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 20:22:08 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
import requests
import os

# In[2]:
def write_comments_in_separeted_files():
    file = open('reviews.txt', 'r').readlines()
    for i, line in enumerate(file):
        newFile = open('reviews/review'+str(i)+'.txt', 'w')
        newFile.write(file[i].split('\t')[0])
        newFile.close()
        
def getFileNames():
    for root, dirs, files in os.walk('./reviews'):
        return files

# In[3]:
def getSenticon():
    listSenticon = []
    mlSenticon = open('MLSenticon.txt', 'r')
    for i in mlSenticon:
        word = i.split('\t')
        listSenticon.append([word[0], word[1].replace('\n', '')])
    return listSenticon
    
# In[4]:
def exists(array, word):
    for i in array:
        if (i[0] == word):
            return i
    return None

# In[5]:
def classification(file, res, senticon):
    file = open(file, 'r')
    points = file[0].split('.')
    if (len(points) > 1):
        # Si tiene puntos evualiar si hay un pero o un no
    else:
        # Si no tiene puntos, evaluar si negativos
    x1 = 0
    x2 = 0
    for r in res:
        for word in r:
            tag = word['tag']
            if (tag[0] == 'A'):
                e = exists(senticon, word['lemma'])
                if (e):
                    if (float(e[1]) > 0):
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


# In[7]:
if __name__ == '__main__':
    write_comments_in_separeted_files()
    senticon = getSenticon()
    res = []
    files = getFileNames()
    for i in files:
        review = './reviews/' + i
        res = obtain_freeling(review)
        classification(review, res, senticon)
    
