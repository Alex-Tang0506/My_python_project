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
                       'http://217.117.13.170:8080', 'http://46.97.0.186:8080', 'http://   104.40.159.62:8118',
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
    def post_source(self, URL):
        self.url = URL
        self.user_agent_random = self.user_agent.random
        self.request_headers = {
            "user-agent": self.user_agent_random
        }

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

        _url_resp = _session.post(self.url, proxies=_proxy, headers=self.request_headers)
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


crawler = MainCrawler()
total_page = 3
f = open("Q2.csv", mode="w", encoding='utf-8-sig', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Date', 'Time', 'Name', 'URL', 'Description', 'AddressInfo', 'StreetAddress', 'PlaceName'])

for i in range(1, total_page):
    url = 'https://www.eventbrite.com/d/ca--palo-alto/business--events/silicon-valley/?page=' + str(i)
    print("Get info from URL: ", url)
    # time.sleep(2)
    resp = crawler.get_source(url)
    print("Read data successful, object:  ", resp)
    # print(resp.text)

    xpath_title_url = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div/section[1]/div[1]/div/ul/li/div/div/div[2]/div/div/div/article/div[1]/div/div/div[1]/a/@href'
                      # '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div/section[1]/div[1]/div/ul/li[2]/div/div/div[2]/div/div/div/article/div[1]/div/div/div[1]/a'
    title_url_list = crawler.get_info_xpath(resp.text, xpath_title_url)
    for title_url in title_url_list:
        print(title_url)
        time.sleep(2)
        resp_sub = crawler.get_source(title_url)

        xpath_date = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[1]/section[1]/div[2]/div[2]/time/p[@data-testid = "event-date"]/text()'
        _date = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_date))

        xpath_time = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[1]/section[1]/div[2]/div[2]/time/p[2]/text()'
        _time = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_time))

        xpath_name = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/div[1]/div/div[2]/div/div[2]/h1[@class = "listing-hero-title"]/text()'
        _name = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_name))

        xpath_description = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[2]/div[2]/div/div/div[@class="eds-text--left"]/p/text()'
        _description = "".join(crawler.get_info_xpath(resp_sub.text, xpath_description))

        xpath_address_info = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[1]/section[@aria-labelledby="location-heading"]/div[@class="event-detail__content"]/p[3]/text()'
        _address_info = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_address_info))
        # if _address_info != "#Null":
        #     _address_info = _address_info.split(",")
        #     _address_locality = _address_info[0]
        #     _address_region_info = _address_info[1].split(" ")
        #     _address_region_info = [none for none in _address_region_info if none != '']
        #     _address_region = _address_region_info[0]
        #     _postalcode = _address_region_info[1]
        # else:
        #     _address_locality = "#Null"
        #     _address_region = "#Null"
        #     _postalcode = "#Null"

        xpath_street_address = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[1]/section[@aria-labelledby="location-heading"]/div[@class="event-detail__content"]/p[2]/text()'
        _street_address = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_street_address))

        xpath_place_name = '//*[@id="root"]/div/div[2]/div/div/div/div[1]/div/main/div/div[2]/div/section[1]/div/div[1]/section[@aria-labelledby="location-heading"]/div[@class="event-detail__content"]/p[1]/text()'
        _place_name = crawler.data_clean(crawler.get_info_xpath(resp_sub.text, xpath_place_name))

        print(_date)
        print(_time)
        print(_name)
        print(_description)
        print(_address_info)
        # print(_address_locality)
        # print(_address_region)
        # print(_postalcode)
        print(_street_address)
        print(_place_name)
        csv_writer.writerow([_date, _time, _name, title_url, _description, _address_info, _street_address, _place_name])
print("All finished")
f.close()

