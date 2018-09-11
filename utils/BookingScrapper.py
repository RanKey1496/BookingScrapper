# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 16:14:58 2018

@author: Jhon GIl Sepulveda
"""

# In[1]:
from selenium import webdriver
from bs4 import BeautifulSoup
import time

# In[2]:
def comments_url(url):
    tab_hashtag = url[-9:]
    print(tab_hashtag)
    if (tab_hashtag != 'hotelTmpl'):
        raise Exception('Por favor ingrese una URL válida de un Hotel en Booking.com')
    return url[:-9] + 'tab-reviews'

# In[3]:
def process_comments(source):
    soup = BeautifulSoup(source, 'html.parser')
    review = []
    review_list = soup.find(class_='review_list')
    review_item_review = review_list.find_all('div', class_='review_item_review')
    for item_review in review_item_review:
        review_neg = item_review.find('p', {'class':'review_neg'})
        review_pos = item_review.find('p', {'class':'review_pos'})
        if (review_neg != None and review_pos != None):
            for div in review_neg.find_all('i'):
                div.decompose()
            for div in review_pos.find_all('i'):
                div.decompose()
            
            review.append({'negative': review_neg.contents[1].replace('\n', ' ').capitalize(),
                           'positive': review_pos.contents[1].replace('\n', ' ').capitalize()})
    return review

# In[4]:
def fetch_comments_all(driver):
    reviews = []
    while (len(reviews) < 50):
        if (len(reviews) == 0):
            reviews.extend(process_comments(driver.page_source))
        else:
            next_page = driver.find_element_by_id('review_next_page_link')
            next_page.click()
            time.sleep(5)
            reviews.extend(process_comments(driver.page_source))
    return reviews

# In[5]:
url = 'https://www.booking.com/hotel/co/art.es.html?label=gen173nr-1FCAEoggJCAlhYSDNYBGgyiAEBmAEKwgEKd2luZG93cyAxMMgBDNgBAegBAfgBC5ICAXmoAgM;sid=57366ce0aedc12300f9c06883f314b39;dest_id=-592318;dest_type=city;dist=0;hapos=12;hpos=12;room1=A%2CA;sb_price_type=total;srepoch=1536528201;srfid=93101b110754dc3526ff0d396a579f7759ef0f09X12;srpvid=6f7c9664584803d5;type=total;ucfs=1&#hotelTmpl'
url = comments_url(url)

driver = webdriver.Chrome()
driver.implicitly_wait(30)
driver.get(url)

# In[6]:
reviews = fetch_comments_all(driver)

# In[7]:
import csv
with open('reviews.txt', 'w', newline='') as file:
    wr = csv.writer(file, delimiter='\t')
    for review in reviews:
        wr.writerow([review['negative'], '0'])
        wr.writerow([review['positive'], '1'])