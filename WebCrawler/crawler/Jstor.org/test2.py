'''
    Request: From the search page to find all subpage URL
'''
import fake_useragent
import requests
import re
import csv
import os
from fake_useragent import UserAgent
import time
from lxml import etree
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import random
import base64
#
# payloads = '{ \
#     "acc": "",\
#     "allowEmptyQuery": false,\
#     "endDate": "",\
#     "filter": "",\
#     "forwardedAdvancedSearchParams": {\
#         "Query": "+George+Eliot",\
#         "so": "rel"\
#     },\
#     "getFlagName": "enable_search_results_infinite_scrolling",\
#     "isAdvancedSearch": false,\
#     "msFacetFields": [],\
#     "pageParams": {},\
#     "referer": "",\
#     "refreqid": "login:7d8404d7bfcc800cdbe619d9ebdaa699",\
#     "searchTerm": "George Eliot",\
#     "sortOrder": "rel",\
#     "startDate": "",\
#     "filterQueries": []\
# }'
#
# def getData(url):
#     _list = []
#     print("正在获取" + url)
#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
#         'Content-Type': 'application/json',
#         "Connection": "close"
#     }
#     session = requests.session()
#     session.keep_alive = False
#     adapter = requests.adapters.HTTPAdapter(max_retries=3)
#     session.mount('https://', adapter)
#     session.mount('http://', adapter)
#
#     try:
#         r = requests.post(url, data=payloads, headers=headers, timeout=10)
#         print(r.text)
#
#     except Exception as e:
#         # 该页面错误
#         print(url)
#         logging.exception(e)
#         return []
#
#
# if __name__ == '__main__':
#     getData('https://www.jstor.org/search-results/grouped-search/')

