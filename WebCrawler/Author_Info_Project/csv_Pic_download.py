import csv
import pandas
import pandas as pd
import urllib.request
import os
import fake_useragent
import time
import requests

user_agent = fake_useragent.UserAgent()


def name_clean(name):
    _str = name
    _list_str = list(_str)
    _spec_symbol = '/'
    if _spec_symbol in _list_str:
        _list_str.pop(_list_str.index(_spec_symbol))
    _index = _list_str.index('(')
    _list_str = _list_str[:_index]
    str_name = ''.join(_list_str)
    return str_name


def getImg(img_url, img_Name):
    _headers = {'User-Agent': user_agent.random}
    _jpg_url = img_url
    r = requests.get(_jpg_url, headers=_headers)
    if r.status_code == '200':
        print(img_url + 'success')
    content = r.content
    with open(r"C:\Users\23619\Desktop/Img/{}".format(img_Name), 'wb') as fp:
        fp.write(content)
        fp.close()


data = pd.read_csv('D:\Tool\PythonProject\WebCrawler\Author_Info_Project\AL_AuthorInfo.csv', usecols=[0, 1])
file_path = 'D:\Tool\PythonProject\WebCrawler\Author_Info_Project\Author_Portrait'
headers = {'User-Agent': user_agent.random}
print(data)
print(dir)

for i in range(data.shape[0]):
    Name, Portrait = data.iloc[i]
    print(Name)
    print(Portrait)
    Name = name_clean(Name)
    getImg(Portrait, Name + '.jpg')
    time.sleep(5)

print('over')
