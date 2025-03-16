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


class MainCrawler:
    def __init__(self, SleepTime=time.sleep(2), NullValue='#Null'):
        self.time_sleep = SleepTime
        self.null_value = NullValue
        self.user_agent = UserAgent()
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

        _proxiesSet = ['http://110.77.134.112:8080', 'http://178.19.97.1:8088', 'http://71.41.27.245:8080', 'http://176.106.120.82:8080',
                       'http://190.104.170.234:8080', 'http://203.160.163.58:8080', 'http://75.134.84.195:8080', 'http://182.180.154.47:3128',
                       'http://23.95.43.171:3128', 'http://82.192.30.238:8080', 'http://198.23.143.27:8080','http://190.186.42.124:8080',
                       'http://36.66.34.10:8080','http://84.22.46.25:8080','http://103.19.109.186:8088','http://61.90.41.97:8888',
                       'http://176.123.129.14:8080','http://46.39.192.48:8080','http://103.247.101.102:8080','http://103.4.66.45:8080',
                       'http://202.56.160.110:8080','http://186.19.175.6:3128','http://103.246.1.101:8080','http://119.18.153.18:8080',
                       'http://125.209.73.57:8080','http://186.101.113.234:8080','http://36.66.242.10:8080','http://180.250.100.66:3128',
                       'http://94.154.222.95:8080','http://80.188.135.10:8080','http://83.241.46.175:8080','http://190.77.134.96:8080',
                       'http://217.117.13.170:8080','http://46.97.0.186:8080','http://	104.40.159.62:8118','http://80.79.127.140:8080',
                       'http://111.68.123.214:8080','http://112.105.11.196:8080']

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

        _proxiesSet = ['http://110.77.134.112:8080', 'http://178.19.97.1:8088', 'http://71.41.27.245:8080', 'http://176.106.120.82:8080',
                       'http://190.104.170.234:8080', 'http://203.160.163.58:8080', 'http://75.134.84.195:8080', 'http://182.180.154.47:3128',
                       'http://23.95.43.171:3128', 'http://82.192.30.238:8080', 'http://198.23.143.27:8080','http://190.186.42.124:8080',
                       'http://36.66.34.10:8080','http://84.22.46.25:8080','http://103.19.109.186:8088','http://61.90.41.97:8888',
                       'http://176.123.129.14:8080','http://46.39.192.48:8080','http://103.247.101.102:8080','http://103.4.66.45:8080',
                       'http://202.56.160.110:8080','http://186.19.175.6:3128','http://103.246.1.101:8080','http://119.18.153.18:8080',
                       'http://125.209.73.57:8080','http://186.101.113.234:8080','http://36.66.242.10:8080','http://180.250.100.66:3128',
                       'http://94.154.222.95:8080','http://80.188.135.10:8080','http://83.241.46.175:8080','http://190.77.134.96:8080',
                       'http://217.117.13.170:8080','http://46.97.0.186:8080','http://104.40.159.62:8118','http://80.79.127.140:8080',
                       'http://111.68.123.214:8080','http://112.105.11.196:8080']

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


url_1 = "https://apps.lib.ua.edu/blogs/this-goodly-land/author-list/"
crawler = MainCrawler()

'''<---------------- 写入csv文件 ---------------->'''
# if __name__ == "__main__":
# 获取当前路径
# current_path = os.getcwd()
# print(current_path)
# f = open(current_path + "\\Archive.org\\narrow_search.csv", mode="w", encoding='utf-8-sig', newline='')
# csv_writer = csv.writer(f)
# csv_writer.writerow(['Subpage_title', 'title_URL'])
# csv_writer.writerow(['Title_URL'])

# for page_num in range(1, 5):
# 1. narrow_search_1 对应文档
# url = "https://archive.org/search.php?query=%28%22George+Eliot%22%29+AND+title%3A%28%22George+Eliot%22%29+AND+-creator%3A%28%22George+Eliot%22%29+AND+date%3A%5B1990-01-01+TO+2022-01-01%5D&page=" + f"{page_num}"

# 2. narrow_search_2 对应文档
# url = "https://archive.org/search.php?query=%28%22George+Eliot%22%29+AND+title%3A%28%22George+Eliot%22%29+AND+mediatype%3A%28texts%29+AND+date%3A%5B1990-01-01+TO+2022-01-01%5D&page=" + f"{page_num}"

# 3. narrow_search_3 对应文档
# url = "https://archive.org/search.php?query=%28%22George+Eliot%22%29+AND+-creator%3A%28%22George+Eliot%22%29+AND+mediatype%3A%28texts%29+AND+date%3A%5B1990-01-01+TO+2022-01-01%5D&page=" + f"{page_num}"

# 4. narrow_search_4 对应文档
# url = "https://archive.org/search.php?query=%28%22George+Eliot%22%29+AND+date%3A%5B1990-01-01+TO+2022-01-01%5D&page=" + f"{page_num}"

#     resp = crawler.get_source(url)
#
#     # 获取div[class = 'C234'] 下的文件， 获取链接和标题
#     xpath_title = '//div[@class = "C234"]/div/a/@title'
#     time.sleep(1)
#     xpath_url = '//div[@class = "C234"]/div/a/@href'
#     time.sleep(1)
#
#     # 相对路径,列表形式存储
#     subpage_url = crawler.get_info_xpath(resp.text, xpath_url)
#     # 绝对路径,列表形式存储
#     subpage_url_abs = crawler.url_transform(crawler.main_address, subpage_url)
#     # 标题,列表形式存储
#     subpage_title = crawler.get_info_xpath(resp.text, xpath_title)
#     time.sleep(1)
#
#     # print(len(subpage_url_abs))
#     # print(len(subpage_title))
#     print(subpage_url_abs)
#     print(subpage_title)
#     for i in range(len(subpage_url_abs)):
#         csv_writer.writerow([subpage_url_abs[i]])
# f.close()
# print('Finish')
