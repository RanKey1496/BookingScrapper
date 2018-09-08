# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 23:02:05 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import re

# In[2]:
main_page = 'https://www.booking.com/hotel/co/jardin-de-laureles.es.html?label=gen173nr-1FCAEoggJCAlhYSDNYBGgyiAEBmAEKwgEKd2luZG93cyAxMMgBDNgBAegBAfgBC5ICAXmoAgM;sid=57366ce0aedc12300f9c06883f314b39;dest_id=-592318;dest_type=city;dist=0;hapos=6;hpos=6;room1=A%2CA;sb_price_type=total;srepoch=1536376113;srfid=e6d219e1570e95cf3b7d9a4361034b9fa968dabcX6;srpvid=114e16180c41002b;type=total;ucfs=1&#hotelTmpl'
tab_hashtag = main_page[-9:]
print(tab_hashtag)
if (tab_hashtag != 'hotelTmpl'):
    raise Exception('Por favor ingrese una URL válida de un Hotel en Booking.com')
main_url = main_page[:-9] + 'tab-reviews'

# In[3]:
page = requests.get(main_url)
print(page)

# In[4]:
soup = BeautifulSoup(page.text, "html.parser")
print(soup)

# In[5]:
def remove_special_characters(text):
    return [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]

# In[6]:
review = []
review_list = soup.find(class_='review_list')
review_item_reviewer = review_list.find_all('div', {'class': 'review_item_reviewer'})
review_item_review = review_list.find_all('div', {'class': 'review_item_review'})

for item_review in review_item_review:
    review_neg = item_review.find('p', {'class':'review_neg'})
    review_pos = item_review.find('p', {'class':'review_pos'})
    if (review_neg != None and review_pos != None):
        for div in review_neg.find_all('i'):
            div.decompose()
        for div in review_pos.find_all('i'):
            div.decompose()
        
        review.append({'negative': remove_special_characters(review_neg.contents[1]),
                       'positive': remove_special_characters(review_pos.contents[1])})
# review_score_badge = review_item_review.find_all('span', {'class': 'review-score-badge'})
# review_item_header_content = review_item_review.find_all('div', {'class': 'review_item_header_content'})
# review_neg = review_item_review.find('p', {'class':'review_neg'})
# review_pos = review_item_review.find_all('p', {'class':'review_pos'})

# print(review_neg)
        
# In[6]:
print(review)