'''
    Request: From the search page to find all subpage URL
'''
import requests
import re
import csv
import os
from fake_useragent import UserAgent
import time
from lxml import etree
from lxml.html.clean import Cleaner
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import random
import base64
import itertools as iter

cleaner = Cleaner()


class MainCrawler:
    def __init__(self, SleepTime=time.sleep(2), NullValue='#Null'):
        self.time_sleep = SleepTime
        self.null_value = NullValue
        self.user_agent = UserAgent(use_cache_server=False)
        self.main_address = 'archive.org'

    def get_source(self, URL):
        self.url = URL
        self.user_agent_random = self.user_agent.random
        self.request_headers = {
            "user-agent": self.user_agent_random
        }
        # _proxiesSet = ['HTTP://60.13.42.120:9999', 'HTTP://163.204.244.207:9999', 'HTTP://163.204.244.207:9999',
        #                'HTTP://125.117.134.99:9000', 'HTTP://123.169.114.81:9999', 'HTTP://58.253.159.230:9999',
        #                'HTTP://182.46.99.247:9999', 'HTTP://171.15.48.235:9999', 'HTTP://120.83.105.226:9999',
        #                'HTTP://113.124.95.210:9999', 'HTTP://171.13.103.241:9999', 'HTTP://123.163.96.42:9999',
        #                'HTTP://27.42.168.46:49345', 'HTTP://122.234.27.22:9000', 'HTTP://125.108.73.47:9000',
        #                'HTTP://125.108.97.209:9000', 'HTTP://113.124.85.24:9999', 'HTTP://27.220.51.228:9000',
        #                'HTTP://113.124.84.205:9999', 'HTTP://110.243.31.147:9999'
        #                ]

        _proxiesSet = ['http://110.77.134.112:8080', 'http://178.19.97.1:8088', 'http://71.41.27.245:8080',
                       'http://176.106.120.82:8080',
                       'http://190.104.170.234:8080', 'http://203.160.163.58:8080', 'http://75.134.84.195:8080',
                       'http://182.180.154.47:3128',
                       'http://23.95.43.171:3128', 'http://82.192.30.238:8080', 'http://198.23.143.27:8080',
                       'http://190.186.42.124:8080',
                       'http://36.66.34.10:8080', 'http://84.22.46.25:8080', 'http://103.19.109.186:8088',
                       'http://61.90.41.97:8888',
                       'http://176.123.129.14:8080', 'http://46.39.192.48:8080', 'http://103.247.101.102:8080',
                       'http://103.4.66.45:8080',
                       'http://202.56.160.110:8080', 'http://186.19.175.6:3128', 'http://103.246.1.101:8080',
                       'http://119.18.153.18:8080',
                       'http://125.209.73.57:8080', 'http://186.101.113.234:8080', 'http://36.66.242.10:8080',
                       'http://180.250.100.66:3128',
                       'http://94.154.222.95:8080', 'http://80.188.135.10:8080', 'http://83.241.46.175:8080',
                       'http://190.77.134.96:8080',
                       'http://217.117.13.170:8080', 'http://46.97.0.186:8080', 'http://	104.40.159.62:8118',
                       'http://80.79.127.140:8080',
                       'http://111.68.123.214:8080', 'http://112.105.11.196:8080']

        _proxy = {'http': random.choice(_proxiesSet)}
        _session = requests.Session()
        _retry = Retry(connect=5, backoff_factor=0.5)
        _adapter = HTTPAdapter(max_retries=_retry)
        _session.mount('http://', _adapter)
        _session.mount('https://', _adapter)

        _url_resp = _session.get(self.url, proxies=_proxy, headers=self.request_headers)
        _url_resp.encoding = 'utf-8'
        return _url_resp

    # 异步加载
    def post_source(self, URL, Headers, Parameters):
        self.url = URL
        self.headers = Headers
        self.parameters = Parameters
        self.user_agent_random = self.user_agent.random
        self.request_headers = {}
        self.user_agent = {"user-agent": self.user_agent_random}
        self.request_headers.update(self.headers)
        self.request_headers.update(self.user_agent)

        # _proxiesSet = ['HTTP://60.13.42.120:9999', 'HTTP://163.204.244.207:9999', 'HTTP://113.121.39.121:9999',
        #                'HTTP://125.117.134.99:9000', 'HTTP://123.169.114.81:9999', 'HTTP://58.253.159.230:9999',
        #                'HTTP://182.46.99.247:9999', 'HTTP://171.15.48.235:9999', 'HTTP://120.83.105.226:9999',
        #                'HTTP://113.124.95.210:9999', 'HTTP://171.13.103.241:9999', 'HTTP://123.163.96.42:9999',
        #                'HTTP://27.42.168.46:49345', 'HTTP://122.234.27.22:9000', 'HTTP://125.108.73.47:9000',
        #                'HTTP://125.108.97.209:9000', 'HTTP://113.124.85.24:9999', 'HTTP://27.220.51.228:9000',
        #                'HTTP://113.124.84.205:9999', 'HTTP://110.243.31.147:9999']

        _proxiesSet = ['http://110.77.134.112:8080', 'http://178.19.97.1:8088', 'http://71.41.27.245:8080',
                       'http://176.106.120.82:8080',
                       'http://190.104.170.234:8080', 'http://203.160.163.58:8080', 'http://75.134.84.195:8080',
                       'http://182.180.154.47:3128',
                       'http://23.95.43.171:3128', 'http://82.192.30.238:8080', 'http://198.23.143.27:8080',
                       'http://190.186.42.124:8080',
                       'http://36.66.34.10:8080', 'http://84.22.46.25:8080', 'http://103.19.109.186:8088',
                       'http://61.90.41.97:8888',
                       'http://176.123.129.14:8080', 'http://46.39.192.48:8080', 'http://103.247.101.102:8080',
                       'http://103.4.66.45:8080',
                       'http://202.56.160.110:8080', 'http://186.19.175.6:3128', 'http://103.246.1.101:8080',
                       'http://119.18.153.18:8080',
                       'http://125.209.73.57:8080', 'http://186.101.113.234:8080', 'http://36.66.242.10:8080',
                       'http://180.250.100.66:3128',
                       'http://94.154.222.95:8080', 'http://80.188.135.10:8080', 'http://83.241.46.175:8080',
                       'http://190.77.134.96:8080',
                       'http://217.117.13.170:8080', 'http://46.97.0.186:8080', 'http://104.40.159.62:8118',
                       'http://80.79.127.140:8080',
                       'http://111.68.123.214:8080', 'http://112.105.11.196:8080']

        # _proxiesSet = ['http://176.106.120.82:8080']

        _proxy = {'http': random.choice(_proxiesSet)}
        _session = requests.Session()
        _retry = Retry(connect=5, backoff_factor=0.5)
        _adapter = HTTPAdapter(max_retries=_retry)
        _session.mount('http://', _adapter)
        _session.mount('https://', _adapter)

        _url_resp = _session.post(self.url, headers=self.request_headers, data=self.parameters, proxies=_proxy)
        _url_resp.encoding = 'utf-8'
        return _url_resp

    def get_info_re(self, Main_Page, Regular_Expression):
        # Format: Regular_Expression = r'<a href="(?P<author_url>.*?)">.*?>'
        self.re = Regular_Expression
        _result = self.re.findall(Main_Page)
        return _result

    def get_info_xpath(self, Main_Page, Xpath_route):
        # etree.strip_tags(Main_Page, ['em'])
        # etree.strip_tags(Main_Page, ['br'])
        # self.xpath_MainPage = etree.HTML(Main_Page,etree.HTMLParser())
        # _result = self.xpath_MainPage.xpath(Xpath_route)
        # self.xpath_MainPage = etree.HTML(Main_Page, etree.HTMLParser())
        self.xpath_MainPage = etree.HTML(Main_Page, etree.HTMLParser())
        _result = self.xpath_MainPage.xpath(Xpath_route)
        return _result

    # def get_title_xpath(self, Main_Page, Xpath_route):
    #     self.xpath_MainPage = etree.HTML(Main_Page, etree.HTMLParser())
    #     _subpage_title = self.xpath_MainPage.xpath(Xpath_route)
    #     return _subpage_title

    def url_transform(self, main_address, list_url):
        for i in range(0, len(list_url)):
            list_url[i] = main_address + list_url[i]
        return list_url

    def base64(self, string_content):
        _s = string_content
        _s = _s.encode()
        _s = base64.b64encode(_s)
        _s = _s.decode('utf-8')
        return _s

    def data_clean(self, list):
        null_value = '#Null'
        _list = list
        if _list == []:
            _list = null_value
        else:
            _list = re.sub('\r\n', '', _list[0])
            _list = ''.join(_list)
        return _list

    def cleanHTML(self, raw_html):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext


url_1 = "https://www.lib.ua.edu/Alabama_Authors/?cat=3&paged=1"
crawler = MainCrawler()
resp = crawler.get_source(url_1)
print("Read data successful, object:  ", resp)

'''<---------------- 写入csv文件 ---------------->'''
if __name__ == "__main__":
    total_page = 40
    null_value = "#Null"
    f = open("AL_AuthorInfo_Name.csv", mode="w", encoding='utf-8-sig', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Author_Name', 'Author_Biography', 'Source', 'Publications'])

    for i in range(1, total_page):
        url = "https://www.lib.ua.edu/Alabama_Authors/?cat=3&paged=" + str(i)
        print("Get info from URL: ", url)

        time.sleep(1)

        resp = crawler.get_source(url)

        time.sleep(1)
        print("URL: " + url + "  info has been retrieved  ", resp)

        # xpath_author_Biography = '//*[@id="content"]/div[@class="post"]/div[@class="entry"]/p[1][contains(strong,
        # "Biography")]/text() | //*[@id="content"]/div[@class="post"]/div[@class="entry"]/p[2][not(contains(strong,
        # "Source"))]/text() | //*[@id="content"]/div[@class="post"]/div[@class="entry"]/p[1][not(strong)]/text()'
        # #'//*[@id="content"]/div[@class="post"]/div[@class="entry"]/p[contains(strong,
        # "Biography:")]/following-sibling::p[1]/text()' # '//*[@id="content"]/div[@class="post"]/div[
        # @class="entry"]/p[1]/text()' # '//*[@id="content"]/div[@class="post"]/div[@class="entry"]/p[contains(
        # strong, "Biography:")]/text()'
        #

        '''1. author name'''
        xpath_author_name = '//*[@id="content"]/div[@class="post"]/h2/a/text()'
        author_name = crawler.get_info_xpath(resp.text, xpath_author_name)

        '''2. author biography'''
        k = 0
        print(author_name)
        for j in range(0, len(author_name)):
            k += 1
            name_start = str(author_name[j])

            # Regex
            re_author_biography = re.compile(r'<a href=.*?' + name_start + '.*?</h2>.*?<div class="entry">.*?<p>.*?Biography.*?(?P<author_Biography>.*?)<strong>', re.S)
            re_author_source = re.compile(r'<a href=.*?' + name_start + '.*?</h2>.*?<div class="entry">.*?<p>.*?Source.*?(?P<author_source>.*?)<strong>', re.S)
            re_author_publications = re.compile(r'<a href=.*?' + name_start + '.*?</h2>.*?<div class="entry">.*?<p>.*?Publication.*?(?P<author_publications>.*?)<p class="postmetadata">', re.S)
            re_author_editor = re.compile(r'<a href=.*?' + name_start + '.*?</h2>.*?<div class="entry">.*?<p>.*?Editor.*?(?P<author_editor>.*?)<p class="postmetadata">', re.S)

            # Get information
            Author_Biography = crawler.data_clean(crawler.get_info_re(resp.text, re_author_biography))
            Author_Source = crawler.data_clean(crawler.get_info_re(resp.text, re_author_source))
            Author_Publications = crawler.data_clean(crawler.get_info_re(resp.text, re_author_publications))

            print(str(k) + ". " + name_start)
            print(Author_Publications)
            print("*******************")
            print("                   ")
            csv_writer.writerow([name_start, Author_Biography, Author_Source, Author_Publications])


        '''3. Source '''
        # re_author_source = re.compile(r'<strong>.*?Source.*?</strong>.*?(?P<author_source>.*?)<strong>Publication', re.S)
        # author_source = crawler.get_info_re(resp.text, re_author_source)

        # for j in author_name:
        #     j = list(j)
        #     result = ''.join(j)
        #     csv_writer.writerow([result])

        print("Scraping from page: {page_num} ----- complete".format(page_num=i))
        time.sleep(2)
        # print(author_Biography)
        print("********************************")
        # print("Length of author name list: {length_name}".format(length_name=len(author_name)))
        # print("Length of Biography list: {length_Bio}".format(length_Bio=len(author_Biography)))
    f.close()
    print("Finished!")
        # print(len(author_Biography))
        # print(len(author_source))
        # print(type(resp.text))
