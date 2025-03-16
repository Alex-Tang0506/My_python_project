"""
This function is focus on extract valid proxy IP address information from different IP webpage
Author: Xianming Tang
Date: 03/05/2022
Models: crawler_web, IP_test, IP_tool, IP_get
"""
import json
import logging
import requests
import time
import re
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, wait
import threading
import random
from fake_useragent import UserAgent


user_agent = UserAgent()

ip_regx = "(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
port_regx = "\d+"


# print(random_user_agent)

def getIplist_kx():
    _list = []
    _list_ip_port = []
    _random_user_agent = user_agent.random
    url_list = [
        'http://www.kxdaili.com/dailiip/1/',
        'http://www.kxdaili.com/dailiip/2/'
    ]
    for u in url_list:
        for i in range(1, 11):  # total 10 pages
            time.sleep(0.5)
            url = u + str(i) + ".html"
            print("Scraping data from -- " + url)
            headers = {
                "User-Agent": _random_user_agent,
                "Connection": "close"
            }
            session = requests.session()
            session.keep_alive = False
            adapter = requests.adapters.HTTPAdapter(max_retries=3)
            session.mount('https://', adapter)
            session.mount('http://', adapter)

            try:
                r = requests.get(url, headers=headers, timeout=3)

                html = etree.HTML(r.content.decode("utf-8"))

                ip_table = html.xpath("//table[@class='active']/tbody/tr")
                if not ip_table:
                    print('Total IP：' + str(len(_list)))
                    break
                for t in ip_table:

                    ip_str = str(t.xpath("./td[1]/text()")).strip()
                    ip = re.findall(ip_regx, ip_str)[0]

                    port_str = str(t.xpath("./td[2]/text()")).strip()
                    port = re.findall(port_regx, port_str)[0]

                    _proxy = {
                        'ip': ip,
                        'port': port,
                        'status': False,
                        'ping': 0
                    }
                    # print(_proxy)
                    if _proxy not in _list:
                        _list.append(_proxy)

            except Exception as e:
                # Error
                print(url)
                # logging.exception(e)
                return []

    print('Scraping complete, total number of IP：' + str(len(_list)))
    return _list


def getIPlsit_hidenMyName():
    _list = []
    _random_user_agent = user_agent.random
    time.sleep(0.5)
    _headers = {
        "User-Agent": _random_user_agent,
        "Connection": "close"
    }
    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    for i in range(0, 5):
        _page = i * 64
        _url = 'https://hidemy.name/en/proxy-list/?start=' + str(_page) + '#list'
        print('=== Scraping data from: ' + _url + ' page: ' + str(i+1) + ' ===')
        time.sleep(0.5)
        try:
            _r = requests.get(_url, headers=_headers, timeout=3)

            _html = etree.HTML(_r.content.decode("utf-8"))

            _ip_list = _html.xpath("/html/body/div[1]/div[4]/div/div[4]/table/tbody/tr/td[1]/text()")
            _ip_port = _html.xpath("/html/body/div[1]/div[4]/div/div[4]/table/tbody/tr/td[2]/text()")
            _ip_type = _html.xpath("/html/body/div[1]/div[4]/div/div[4]/table/tbody/tr/td[5]/text()")
            _ip_speed = _html.xpath("/html/body/div[1]/div[4]/div/div[4]/table/tbody/tr/td[4]/div/p/text()")
            _ip_anonymity = _html.xpath("/html/body/div[1]/div[4]/div/div[4]/table/tbody/tr/td[6]/text()")
            if not _ip_list:
                print('Total IP：' + str(len(_list)))
                break

            for j in range(0, len(_ip_list)):
                _ip_speed[j] = int(_ip_speed[j][:-3])
                if _ip_type[j] != 'HTTP':
                    continue
                else:
                    _proxy = {
                        'ip': _ip_list[j],
                        'port': _ip_port[j],
                        'status': False,
                        'ping': 0
                    }
                    if _proxy not in _list:
                        _list.append(_proxy)
        except Exception as e:
            # Error
            print(_url)
            # logging.exception(e)
            return []
    print('Scraping complete, total number of IP：' + str(len(_list)))
    return _list


'''Test IP is valid or not'''


def proxy_test(ip, port):
    _random_user_agent = user_agent.random
    headers = {
        "User-Agent": _random_user_agent,
        "Connection": "close"
    }
    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=1)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    proxy_txt = {"http": ip + ":" + port,
                 "https": ip + ":" + port}

    _pay_load = '{ \
        "acc": "",\
        "allowEmptyQuery": false,\
        "endDate": "",\
        "filter": "",\
        "forwardedAdvancedSearchParams": {\
            "Query": "+George+Eliot",\
            "so": "rel"\
        },\
        "getFlagName": "enable_search_results_infinite_scrolling",\
        "isAdvancedSearch": false,\
        "msFacetFields": [],\
        "pageParams": {},\
        "referer": "",\
        "refreqid": "login:7d8404d7bfcc800cdbe619d9ebdaa699",\
        "searchTerm": "George Eliot",\
        "sortOrder": "rel",\
        "startDate": "",\
        "filterQueries": []\
    }'

    try:
        req = requests.post('https://www.jstor.org/search-results/grouped-search/', proxies=proxy_txt,
                           headers=headers, data=_pay_load,
                           timeout=1)
        time.sleep(0.5)
        # print(req.text)
    except Exception as e:
        # print("Fail:", proxy_txt)
        # logging.exception(e)
        return False
    return True


'''Filter the valid IP'''


def proxies_filter(_list):
    print('Testing valid IP...')
    time.sleep(0.5)
    i = 0
    while i < len(_list):
        _proxy = _list[i]
        _ts = time.time()
        result = proxy_test(_proxy['ip'], _proxy['port'])

        if result:
            _te = time.time()
            _proxy['status'] = result
            _proxy['ping'] = int((_te - _ts) * 1000)
            # print("SUCCESS:", _proxy)
            i = i + 1
        else:
            # print("FAIL:", _proxy)
            _list.pop(i)
        print("===", i, "/", len(_list), "===")
    print("Testing complete, total valid IP is:", len(_list))


def check_proxy(_proxy, _temp_list, _desc):
    _ts = time.time()
    result = proxy_test(_proxy['ip'], _proxy['port'])
    if result:
        _te = time.time()
        _proxy['status'] = result
        _proxy['ping'] = int((_te - _ts) * 1000)
        _temp_list.append(_proxy)
        # print("SUCCESS:", _proxy)
    print(_desc)


'''Multi-thread filter'''
threadPool = ThreadPoolExecutor(100)


def proxies_filter_multiThread(_list):
    _temp_list = []
    _thread_list = []
    i = 0
    for _proxy in _list:
        _ts = time.time()
        _thread = threadPool.submit(check_proxy, _proxy, _temp_list,
                                    str("===" + str(i) + "/" + str(len(_list)) + "==="))
        _thread_list.append(_thread)
        i += 1
    wait(_thread_list)
    _list[:] = _temp_list  # replace new list
    print("Testing complete,total valid IP is:", len(_list))


'''KX Ip proxy results'''
# dic_proxy_kx = getIplist_kx()
# print(dic_proxy_kx)
# print(proxies_filter(dic_proxy_kx))

'''HidenMyName IP proxy results'''
dic_proxy_hidenMyName = getIPlsit_hidenMyName()
print(dic_proxy_hidenMyName)
print(proxies_filter(dic_proxy_hidenMyName))
