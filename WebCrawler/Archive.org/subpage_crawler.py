import sys
sys.path.append(r'/')
from Main_Crawler import MainCrawler
import csv
import os
import pandas as pd
import numpy as np
import time
import re

null_value = '#Null'


def data_clean(list):
    _list = list
    if _list == []:
        _list = null_value
    else:
        _list = re.sub('\r\n', '', _list[0])
        _list = ''.join(_list)
    return _list


# Initialize
maincrawler = MainCrawler()
# subpage_Main function

# 文件预处理(pre-process)
current_path = os.getcwd()
csvFile = pd.read_csv(current_path + "\\Archive.org\\narrow_search.csv", header=None, sep=',')
csvFile_np = np.array(csvFile).flatten()
csvFile_list = csvFile_np.tolist()
print(csvFile_list)

# Read url and get information
if __name__ == "__main__":
    # Create a csv file
    f = open(current_path + "\\Archive.org\\narrow_search_4.csv", mode="w", encoding='utf-8-sig', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Pub_date', 'Topics', 'Title', 'Author', 'Publisher', 'Language', 'Description',
                        'Access_restricted', 'Added_date', 'Pdf_link', 'TXT_link', 'ISBN'])

    length_csvFileList = len(csvFile_list)
    '''
    Features list
    1. Publication date
    2. Topics
    3. Publisher
    4. Collection
    5. Language
    6. Description
    7. Access_restricted
    8. Added date
    9. Pdf link
    10. TXT link
    11. ISBN
    '''

    for i in range(1, length_csvFileList):
        # url
        url = 'http://' + csvFile_list[i]
        resp = maincrawler.get_source(url)

        '''x-path route (According to the features provided above,related path)'''
        xpath_pub_date = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[contains(dt,"Publication date")]/dd/a/span/text()'
        xpath_topics = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[contains(dt,"Topics")]/dd/a/text()'
        xpath_title = '//*[@id="maincontent"]/div[4]/div/div/div[2]/h1[contains(@class,"item-title")]/span/text()'
        xpath_author = '//*[@id="maincontent"]/div[4]/div/div/div[2]/dl[contains(dt,"by")]/dd/span/a/text()'
        xpath_publisher = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[contains(dt,"Publisher")]/dd/span/text()'
        # xpath_collection = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[4]/dd/a'
        xpath_language = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[contains(dt,"Language")]/dd/span/a/text()'
        xpath_description = '//*[@id="descript"]/text()[2]'
        xpath_access_res = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[5]/dl[contains(dt,"Access-restricted-item")]/dd/text()'
        xpath_added_date = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[5]/dl[contains(dt,"Addeddate")]/dd/text()'
        xpath_pdf_link = '//*[@id="maincontent"]/div[5]/div/div/div[2]/section[2]/div[7]/a/@href'
        xpath_txt_link = '//*[@id="maincontent"]/div[5]/div/div/div[2]/section[2]/div[4]/a/@href'
        xpath_ISBN = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[5]/dl[12]/dd/span/text()'

        ''' x-path route (Absolute path)'''
        # xpath_pub_date = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[2]/dl[1]/dd/a/span/text()'
        # xpath_topics = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[2]/dl[2]/dd/a/text()'
        # xpath_publisher = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[2]/dl[3]/dd/span/text()'
        # # xpath_collection = '//*[@id="maincontent"]/div[5]/div/div/div[1]/div[2]/dl[4]/dd/a'
        # xpath_language = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[2]/dl[7]/dd/span/a/text()'
        # xpath_description = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[4]/text()[2]'
        # xpath_access_res = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[5]/dl[1]/dd/text()'
        # xpath_added_date = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[5]/dl[2]/dd/text()'
        # xpath_pdf_link = '/html/body/div[1]/main/div[5]/div/div/div[2]/section[2]/div[7]/a/@href'
        # xpath_txt_link = '/html/body/div[1]/main/div[5]/div/div/div[2]/section[2]/div[4]/a/@href'
        # xpath_ISBN = '/html/body/div[1]/main/div[5]/div/div/div[1]/div[5]/dl[12]/dd/span/text()'



        '1. Pub data'
        pub_date = data_clean(maincrawler.get_info_xpath(resp.text, xpath_pub_date))
        # pub_date_list.append(pub_date)

        '2. topics'
        topics = data_clean(maincrawler.get_info_xpath(resp.text, xpath_topics))
        # topics_list.append(topics)

        '3. title'
        title = data_clean(maincrawler.get_info_xpath(resp.text, xpath_title))

        '4. author'
        author = data_clean(maincrawler.get_info_xpath(resp.text, xpath_author))

        '5. publisher'
        publisher = data_clean(maincrawler.get_info_xpath(resp.text, xpath_publisher))
        # publisher_list.append(publisher)

        '6. collection'
        # collection = data_clean(maincrawler.get_info_xpath(resp.text, xpath_collection))
        # print(collection)

        '7. language'
        language = data_clean(maincrawler.get_info_xpath(resp.text, xpath_language))
        # language_list.append(language)

        '8. description'
        description = data_clean(maincrawler.get_info_xpath(resp.text, xpath_description))
        # description_list.append(description)

        '9. access_restricted'
        access_restricted = data_clean(maincrawler.get_info_xpath(resp.text, xpath_access_res))
        # accRes_list.append(access_restricted)

        '10. added date'
        added_date = data_clean(maincrawler.get_info_xpath(resp.text, xpath_added_date))
        # addedDate_list.append(added_date)

        '11. pdf link'
        pdf_link = data_clean(maincrawler.get_info_xpath(resp.text, xpath_pdf_link))
        # pdf_link_list.append(pdf_link)

        '12. txt link'
        txt_link = data_clean(maincrawler.get_info_xpath(resp.text, xpath_txt_link))
        # txt_link_list.append(txt_link)

        '13. ISBN'
        ISBN = maincrawler.get_info_xpath(resp.text, xpath_ISBN)
        # ISBN_list.append(ISBN)

        csv_writer.writerow(
            [pub_date, topics, title, author, publisher, language, description, access_restricted, added_date, pdf_link, txt_link,
             ISBN])

        time.sleep(1)
    f.close()
print('Over')
