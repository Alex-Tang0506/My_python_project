""" Request: Extract all all the biographies and portraits of the authors on website """

# step.1 : Find the Author's list
# step.2 : Extract each url of Author from the list
# step.3 : Extract all information we needed from webpage

import requests
import re
import csv
import pandas as pd
import numpy
from fake_useragent import UserAgent
import random
import time


user_agent = UserAgent()


def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp


url = "https://apps.lib.ua.edu/blogs/this-goodly-land/author-list/"

main_page = get_source(url)
main_page.encoding = 'utf-8'
main_page_txt = main_page.text
author_url_list = []
# print(author_page.encoding)
# print(author_page.apparent_encoding)
# print(author_page.text)
re_author = re.compile(r'<h2>Author List</h2>.*?(?P<author>.*?)<footer>', re.S)
re_author_url = re.compile(r'<a href="(?P<author_url>.*?)">.*?>', re.S)

re_author_information = re.compile(r'<h2>(?P<name_lifetime>.*?)</h2>.*?'
                                   r'Other Names Used</h3><ul>.*?(?P<other_name>.*?)</ul>.*?', re.S)
                                   # r'<h3 class="author-h3">Alabama Connections</h3><ul>.*?(?P<Al_connection>.*?).*</ul>.*?'
                                   # r'<h3 class="author-h3">Selected Works</h3><ul>(?P<selected_works>.*?)</ul>.*?'
                                   # r'<h3 class="author-h3">Literary Awards</h3><ul>(?P<literary_awards>.*?)</ul>.*?'
                                   # r'<h3 class="author-h3">Biographical Information</h3>(?P<biography_info>.*?)<h3 class="author-h3">', re.S)
                                   # r'<h3 class="author-h3">Interests and Themes</h3><p>(?P<interests_and_themes>.*?)</p>.*?'
                                   # r'<h3 class="author-h3">For More Information</h3>(?P<more_info>.*?)<h3 class="author-h3">.*?'
                                   # r'Reference Books</h3><ul>(?P<reference_books>.*?)</ul>.*?'
                                   # r'<h3 class="author-h3">Reference Articles</h3><ul>(?P<reference_articles>.*?)</ul>.*?'
                                   # r'<h3 class="author-h3">Reference Web Sites</h3><ul>(?P<reference_websites>.*?)</ul>.*?'
                                   # r'<h3 class="author-h3">Location of Papers</h3><p><ul>.*?(?P<location_of_papers>.*?)</ul>.*?'

# Subpage information

# Subpage information list
list_author_name_lifetime = []
list_author_BiographicalInformation =[]

f = open("AL_AuthorInfo_all.csv", mode="w", encoding='utf-8-sig', newline='')
csv_writer = csv.writer(f)

author_list = re_author.finditer(main_page_txt)

print(author_list)
for i in author_list:
    main = i.group('author').strip()
    author_url = re_author_url.finditer(main)
    for j in author_url:
        print(j.group('author_url'))
        author_url_list.append(j.group('author_url'))
print(author_url_list)


for i_1 in author_url_list:
    author_page = get_source(i_1)
    author_page.encoding = 'utf-8'
    author_page_txt = author_page.text
    author_information = re_author_information.finditer(author_page_txt)
    for j in author_information:
        print(j.group('name_lifetime'))
        print(j.group('other_name'))
        # print(j.group('Al_connection'))
        # print(j.group('selected_works'))
        # print(j.group('literary_awards'))
        # print(j.group('biography_info'))
        # print(j.group('interests_and_themes'))
        # print(j.group('more_info'))
        # print(j.group('reference_books'))
        # print(j.group('reference_articles'))
        # print(j.group('reference_websites'))
        # print(j.group('location_of_papers'))

        time.sleep(2)

        dic = j.groupdict()
        # dic['name_lifetime'] = dic['name_lifetime'].strip()
        # dic['img'] = dic['img'].strip()
        # dic['other_name'] = dic['other_name'].strip()
        # dic['Al_connection'] = dic['Al_connection'].strip()
        # dic['selected_works'] = dic['selected_works'].strip()
        # dic['literary_awards'] = dic['literary_awards'].strip()
        # dic['biography_info'] = dic['biography_info'].strip()
        # dic['interests_and_themes'] = dic['interests_and_themes'].strip()
        # dic['more_info'] = dic['more_info'].strip()
        # dic['reference_books'] = dic['reference_books'].strip()
        # dic['reference_articles'] = dic['reference_articles'].strip()
        # dic['reference_websites'] = dic['reference_websites'].strip()
        # dic['location_of_papers'] = dic['location_of_papers'].strip()

        csv_writer.writerow(dic.values())
f.close()
print("over")


