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
null_value = '#Null'


def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp


def data_clean(list):
    _list = list
    if _list == []:
        _list = null_value
    else:
        _list = re.sub('\r\n', '', _list[0])
        _list = ''.join(_list)
    return _list


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

re_author_information_name = re.compile(r'<h2>(?P<name_lifetime>.*?)</h2>.*?', re.S)
re_author_portrait = re.compile(r'<img alt="" class="author-img" src="(?P<portrait_link>.*?)".*?', re.S)
re_author_information_otherName = re.compile(r'Other Names Used</h3><ul>.*?(?P<other_name>.*?)</ul>.*?', re.S)
re_author_Al_connections = re.compile(r'Alabama Connections</h3><ul>.*?(?P<Al_connection>.*?)</ul>.*?', re.S)
re_author_selected_works = re.compile(r'Selected Works</h3><ul>.*?(?P<selected_works>.*?)</ul>', re.S)
re_author_literary_awards = re.compile(r'Literary Awards</h3><ul>(?P<literary_awards>.*?)</ul>.*?', re.S)
re_author_bio_info = re.compile(r'<h3 class="author-h3">Biographical Information</h3>(?P<biography_info>.*?)<h3 class="author-h3">', re.S)
re_author_interest_themes = re.compile(r'Interests and Themes</h3>(?P<interests_and_themes>.*?)<h3 class="author-h3">.*?', re.S)
re_author_more_info = re.compile(r'For More Information</h3>(?P<more_info>.*?)<h3 class="author-h3">.*?', re.S)
re_author_reference_books = re.compile(r'Reference Books</h3><ul>(?P<reference_books>.*?)</ul>.*?',re.S)
re_author_reference_articles = re.compile(r'Reference Articles</h3><ul>(?P<reference_articles>.*?)</ul>.*?', re.S)
re_author_reference_websites = re.compile(r'Reference Web Sites</h3><ul>(?P<reference_websites>.*?)</ul>.*?', re.S)
re_author_reference_book_prefaces = re.compile(r'Reference Book Prefaces</h3><ul>(?P<reference_book_prefaces>.*?)</ul>.*?', re.S)
re_author_reference_book_chapters_encyclopedia_entries = re.compile(r'Reference Book Chapters and Encyclopedia Entries</h3><ul>(?P<reference_book_chapters_encyclopedia_entries>.*?)</ul>.*?', re.S)
re_author_location_papers = re.compile(r'Location of Papers</h3><p><ul>.*?(?P<location_of_papers>.*?)</ul>.*?',re.S)



# r'<h3 class="author-h3">Location of Papers</h3><p><ul>.*?(?P<location_of_papers>.*?)</ul>.*?'


# Subpage information

'''------------ Subpage information list------------'''
list_author_name_lifetime = []
list_author_BiographicalInformation = []

'''------------ Generate csv file ------------'''
f = open("AL_AuthorInfo_all.csv", mode="w", encoding='utf-8-sig', newline='')
csv_writer = csv.writer(f)

'''------------Generate all columns------------'''
csv_writer.writerow(['Author_Name', 'Portrait_Link', 'Other_Name', 'Al_Connection', 'Select_Works', 'Literary_Awards',
                     'Bio_Info', 'Interest_Themes', 'More_Info', 'Reference_Books', 'Reference_Articles', 'Reference_Websites',
                     'Reference_Books_Prefaces', 'Reference_Book_Chapters_Encyclopedia_Entries', 'Location_Papers'])

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

    ''' ------------ Assignment 赋值 ------------ '''
    author_name = re_author_information_name.findall(author_page_txt)
    author_portrait = re_author_portrait.findall(author_page_txt)
    author_otherName = re_author_information_otherName.findall(author_page_txt)
    author_Al_connections = re_author_Al_connections.findall(author_page_txt)
    author_selected_works = re_author_selected_works.findall(author_page_txt)
    author_literary_awards = re_author_literary_awards.findall(author_page_txt)
    author_bio_info = re_author_bio_info.findall(author_page_txt)
    author_interest_themes = re_author_interest_themes.findall(author_page_txt)
    author_more_info = re_author_more_info.findall(author_page_txt)
    author_reference_books = re_author_reference_books.findall(author_page_txt)
    author_reference_articles = re_author_reference_articles.findall(author_page_txt)
    author_reference_websites = re_author_reference_websites.findall(author_page_txt)
    author_reference_book_prefaces = re_author_reference_book_prefaces.findall(author_page_txt)
    author_reference_book_chapters_encyclopedia_entries = re_author_reference_book_chapters_encyclopedia_entries.findall(author_page_txt)
    author_location_papers = re_author_location_papers.findall(author_page_txt)

    ''' ------------ Clean 清理 ------------ '''
    # 1. author_name
    author_name = ''.join(author_name)

    # 2. author_portrait
    if author_portrait == []:
        author_portrait = null_value
    else:
        author_portrait = ''.join(author_portrait)

    # 3. author_other_name
    author_otherName = data_clean(author_otherName)

    # 4. author_al_connection
    author_Al_connections = data_clean(author_Al_connections)

    # 5. author_selected_works
    author_selected_works = data_clean(author_selected_works)

    # 6. author_literary_awards
    author_literary_awards = data_clean(author_literary_awards)

    # 7. author_bio_info
    author_bio_info = data_clean(author_bio_info)

    # 8. author_interest_themes
    author_interest_themes = data_clean(author_interest_themes)

    # 9. author_more_info
    author_more_info = data_clean(author_more_info)

    # 10. author_reference_books
    author_reference_books = data_clean(author_reference_books)

    # 11. author_reference_articles
    author_reference_articles = data_clean(author_reference_articles)

    # 12. author_reference_websites
    author_reference_websites = data_clean(author_reference_websites)

    # 13. author_reference_books_prefaces
    author_reference_book_prefaces = data_clean(author_reference_book_prefaces)

    # 14. author_reference_book_chapters_encyclopedia_entries
    author_reference_book_chapters_encyclopedia_entries = data_clean(author_reference_book_chapters_encyclopedia_entries)

    # 15. author_location_papers
    author_location_papers = data_clean(author_location_papers)



    ''' ------------ Print 打印 ------------ '''
    print(author_name)
    # print(author_portrait)
    # print(author_otherName)
    # print(author_Al_connections)
    # print(author_selected_works)
    # print(author_literary_awards)
    # print(author_bio_info)
    # print(author_interest_themes)
    # print(author_more_info)
    # print(author_reference_books)
    # print(author_reference_articles)
    # print(author_reference_websites)
    # print(author_reference_book_prefaces)
    # print(author_reference_book_chapters_encyclopedia_entries)
    print(author_location_papers)

    csv_writer.writerow([author_name, author_portrait, author_otherName, author_Al_connections, author_selected_works, author_literary_awards,
                         author_bio_info, author_interest_themes, author_more_info, author_reference_books, author_reference_articles, author_reference_websites,
                         author_reference_book_prefaces, author_reference_book_chapters_encyclopedia_entries, author_location_papers])

    # csv_writer.writerow(
    #     ['Author_Name', 'Portrait_Link', 'Other_Name', 'Al_Connection', 'Select_Works', 'Literary_Awards',
    #      'Bio_Info', 'Interest_Themes', 'More_Info', 'Reference_Books', 'Reference_Articles', 'Reference_Websites',
    #      'Reference_Books_Prefaces', 'Reference_Book_Chapters_Encyclopedia_Entries', 'Location_Papers'])

    time.sleep(2)
f.close()
print('Over')
