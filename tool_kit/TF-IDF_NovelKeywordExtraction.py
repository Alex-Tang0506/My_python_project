import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rake_nltk import Rake
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
from summa import keywords
from keybert import KeyBERT
# nltk.download()

wordnet = WordNetLemmatizer()
kw_model = KeyBERT(model='all-mpnet-base-v2')
stop_words = stopwords.words("english")
rake_nltk_var = Rake()

Book_list = ['Adam Bede(1859).txt', 'Brother Jacob(1864).txt', 'Daniel Deronda(1876).txt', 'Felix Holt, the Radical(1866).txt', 'Impressions of Theophrastus Such (1879).txt',
             'Middlemarch(1871-72).txt','Romola(1863).txt','Scenes of Clerical Life(1858).txt','Silas Marner(1861).txt','The Lifted Veil(1859).txt','The Mill on the Floss(1860).txt']

# Book_list = ['Brother Jacob(1864).txt']

path = os.getcwd()
print(path)

def clean_text(x):
    # x = x.lower()
    # x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'@\S+', ' ', x)
    x = re.sub(r'#\S+', ' ', x)
    x = re.sub(r'\'\w+', '', x)
    # x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x

def TxtRead(BookList):
    for i in range(0,len(BookList)):
        with open(path + r'\data_GENovels\{}'.format(BookList[i]), 'r') as f:
            content = f.read()
            content = clean_text(content)
            # TR_keywords = keywords.keywords(content, scores=True)
            # print('---------------')
            # print(TR_keywords[:100])

            keywords = kw_model.extract_keywords(content,

                                                 keyphrase_ngram_range=(1,1),

                                                 stop_words='english',

                                                 highlight=False,

                                                 top_n=50)

            keywords_list = list(dict(keywords).keys())
            with open(path + r'\data_GENovels\keywords_{}'.format(BookList[i]), 'w') as f_1:
                f_1.write(str(keywords_list))
            print(keywords_list)
    f.close()
    f_1.close()
TxtRead(Book_list)
# text = """spaCy is an open-source software library for advanced natural language processing,
# written in the programming languages Python and Cython. The library is published under the MIT license
# and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
# print(type(text))
# rake_nltk_var.extract_keywords_from_text(text)
# keyword_extracted = rake_nltk_var.get_ranked_phrases()[:3]
# print(keyword_extracted)
