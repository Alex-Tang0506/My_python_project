import json
import logging
import requests
import time
import re
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, wait
import threading
import random



# 获取代理ip 开心
ip_regx = "(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
port_regx = "\d+"


def getIplist_kx():
    # 正在获取
    _list = []
    url_list = [
        'http://www.kxdaili.com/dailiip/1/',
        'http://www.kxdaili.com/dailiip/2/'
    ]
    for u in url_list:
        for i in range(1, 11):  # 共10页
            time.sleep(0.1)
            url = u + str(i) + ".html#ip"
            print("正在获取" + url)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
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
                    print('共计获取IP：' + str(len(_list)))
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
                    #print(_proxy)
                    if _proxy not in _list:
                        _list.append(_proxy)

            except Exception as e:
                # 该页面错误
                print(url)
                #logging.exception(e)
                return []

    print('获取完成，共计获取IP：' + str(len(_list)))
    return _list


# 获取代理ip 西刺
ip_regx = "(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
port_regx = "\d+"


def getIplist_xici():
    # 正在获取
    _list = []
    index = 1
    end = 21
    for i in range(1, 2):  # 第1页 前20个够了
        time.sleep(0.1)
        url = 'https://www.xicidaili.com/wn/' + str(i)
        print("正在获取" + url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
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
            ip_table = html.xpath("//table[@id='ip_list']/tr")

            while index < end:  # len(ip_table):
                t = ip_table[index]
                index = index + 1
                ip_str = str(t.xpath("./td[2]/text()")).strip()
                ip = re.findall(ip_regx, ip_str)[0]

                port_str = str(t.xpath("./td[3]/text()")).strip()
                port = re.findall(port_regx, port_str)[0]

                _proxy = {
                    'ip': ip,
                    'port': port,
                    'status': False,
                    'ping': 0
                }
                # print(_proxy)
                _list.append(_proxy)

        except Exception as e:
            # 该页面错误
            print(url)
            #logging.exception(e)
            return []

    print('获取完成，共计获取IP：' + str(len(_list)))
    return _list


# 获取代理ip 89
ip_regx = "(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
port_regx = "\d+"


def getIplist_89():
    # 正在获取
    _list = []
    for i in range(1, 50):
        time.sleep(0.1)
        url = 'http://www.89ip.cn/index_' + str(i) + '.html'
        print("正在获取" + url)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36",
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
            ip_table = html.xpath("//table[@class='layui-table']/tbody/tr")
            if not ip_table:
                print('获取完成，共计获取IP：' + str(len(_list)))
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
                #print(_proxy)
                _list.append(_proxy)

        except Exception as e:
            # 该页面错误
            print(url)
            #logging.exception(e)
            # return []

    return _list


# 获取IP列表 x2
def getIplist_x2():
    time.sleep(0.1)
    _list = []
    url = 'http://api.xedl.321194.com/getip?num=200&type=2&port=1&pack=3054&ts=1&cs=0&lb=1'
    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    try:
        r = session.get(url, timeout=3)
        #print(r.text)

        data = json.loads(r.text)
        for p in data['data']:
            _proxy = {
                'ip': p['ip'],
                'port': str(p['port']),
                'status': False,
                'ping': 0
            }
            _list.append(_proxy)

    except Exception as e:
        # 该页面错误
        print(url)
        #logging.exception(e)
        return []
    print('获取完成，共计获取IP：' + str(len(_list)))
    return _list


# 刷新代理IP列表
def update_proxies():
    _list_cache = []
    # _list_cache = _list_cache + getIplist_x2()
    # _list_cache = _list_cache + getIplist_kx()
    _list_cache = _list_cache + getIplist_89()
    _list_cache = _list_cache + getIplist_xici()
    # proxies_filer(_list_cache)
    proxies_filer_more_thread(_list_cache)  # 多线程方式

    _dict = {}
    for p in _list_cache:
        _key = p['ip'] + ':' + p['port']
        _dict[_key] = p

    return _dict


proxy_list = {}


# 使用代理请求URL
def request_by_proxy_post(url, params, headers):

    while len(proxy_list) == 0:
        print("当前无可用代理！等待维护线程补充代理池！")
        time.sleep(1)

    _l = list(proxy_list.values())
    _proxy_item = random.choice(_l)

    _proxy_str = _proxy_item['ip'] + ":" + str(_proxy_item['port'])

    _proxy_txt = {"http": _proxy_str,
                      "https": _proxy_str}

    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    try:

        req = requests.post(url, data=params, proxies=_proxy_txt, headers=headers, timeout=3)
        print(req.text)

    except Exception as e:

        print("Fail:", _proxy_txt)
        # logging.exception(e)
        # 当调用处得知错误后 在外部进行列表修正
        return '', _proxy_str

    return req.text, _proxy_str


# 使用代理请求URL
def request_by_proxy(url, headers, test_proxy=None):
    if test_proxy == None:
        #emp = 0
        while len(proxy_list) == 0:
            #emp = 1
            print("当前无可用代理！等待维护线程补充代理池！")
            #logging.warning('当前无可用代理！' + url)
            # thread_num = len(threading.enumerate())
            # print("线程数量是%d" % thread_num)
            time.sleep(1)

        _l = list(proxy_list.values())
        _proxy_item = random.choice(_l)

        _proxy_str = _proxy_item['ip'] + ":" + str(_proxy_item['port'])

        _proxy_txt = {"http": _proxy_str,
                      "https": _proxy_str}

    else:
        _proxy_txt = test_proxy

    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    try:


        # raise Exception('xxxxxxxxxx')
        req = requests.get(url, proxies=_proxy_txt, headers=headers, timeout=3)
        #print(req.text)
        #if emp == 1:
            #logging.warning('重新请求成功' + url)
            #emp = 0

    except Exception as e:
        #if emp == 1:
            #logging.warning('重新请求失败' + url)
            #emp = 0
        print("Fail:", _proxy_txt)
        # logging.exception(e)
        # 当调用处得知错误后 在外部进行列表修正
        return '', _proxy_str

    return req.text, _proxy_str


# 清除不可用的代理IP
def del_proxy(key):
    global proxy_list

    _p = proxy_list.pop(key,'没有该键')

    print("当前代理失效:", _p, "/ 剩余代理数:", len(proxy_list))

    #  代理IP匮乏 启动一个线程去获取新的代理IP


#   elif(len(proxy_list) < 10):
#       _thread = threading.Thread(target=update_proxies)
#       _thread.start()
#       print('开始采集新的代理列表！')


# 代理列表维护线程
def proxies_update_thread():
    global proxy_list

    try:

        while True:
            print("正在检测可用代理数量，当前：", len(proxy_list))
            # thread_num = len(threading.enumerate())
            # print("线程数量是%d" % thread_num)
            if len(proxy_list) < 10:
                print('开始采集新的代理列表！')
                proxy_list = update_proxies()
                print('代理列表克隆完成！')
            else:
                time.sleep(10)
    except Exception as e:
        logging.exception(e)


# 代理可用性筛选
def proxies_filer(_list):
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
    print("检测完成,共计可用IP:", len(_list))


# 验证代理可用性并添加到临时列表
def cheak_proxy(_proxy, _temp_list, _desc):
    _ts = time.time()
    result = proxy_test(_proxy['ip'], _proxy['port'])
    if result:
        _te = time.time()
        _proxy['status'] = result
        _proxy['ping'] = int((_te - _ts) * 1000)
        _temp_list.append(_proxy)
        #print("SUCCESS:", _proxy)
    print(_desc)


# 多线程检测代理IP
threadPool = ThreadPoolExecutor(100)


def proxies_filer_more_thread(_list):
    _temp_list = []
    _thread_list = []
    i = 0
    for _proxy in _list:
        _ts = time.time()
        _thread = threadPool.submit(cheak_proxy, _proxy, _temp_list,
                                    str("===" + str(i) + "/" + str(len(_list)) + "==="))
        _thread_list.append(_thread)
        i += 1
    wait(_thread_list)
    _list[:] = _temp_list  # 将新列表替换
    print("检测完成,共计可用IP:", len(_list))


# 测试代理可用性
def proxy_test(ip, port):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
        "Connection": "close"
    }
    session = requests.session()
    session.keep_alive = False
    adapter = requests.adapters.HTTPAdapter(max_retries=1)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    proxy_txt = {"http": ip + ":" + port,
                 "https": ip + ":" + port}

    try:
        req = requests.get('https://www.airbnb.cn/api/v2/explore_tabs', proxies=proxy_txt, headers=headers, timeout=1)
        # print(req.text)
    except Exception as e:
        # print("Fail:", proxy_txt)
        # logging.exception(e)
        return False
    return True


# 启动前工作
def init():
    print('代理模块初始化！')
    global proxy_list
    proxy_list = update_proxies()

    # 启动代理维护线程
    update_proxies_thread = threading.Thread(target=proxies_update_thread)
    update_proxies_thread.setDaemon(False)
    update_proxies_thread.start()

    print('代理模块初始化完成！')


MAX_RETRIES = 30
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
    "Connection": "close"
}


def debug_init():
    print('代理模块初始化！')
    global proxy_list
    proxy_list = getIplist_89()
    proxies_filer_more_thread(proxy_list)
    print('代理模块初始化完成！')








if __name__ == '__main__':
    init()
    # proxies_filer_more_thread(_list)

    # update_proxies()

    # for i in range(5):
    #    time.sleep(1)
    #    request_by_proxy('https://www.airbnb.cn/api/v2/explore_tabs?_format=for_explore_search_web&_intents=p1&auto_ib=true&currency=CNY&experiences_per_grid=20&fetch_filters=true&guidebooks_per_grid=20&has_zero_guest_treatment=true&is_guided_search=true&is_new_cards_experiment=true&is_standard_search=true&items_per_grid=50&items_offset=0&key=d306zoyjsyarp7ifhu67rjxn52tv0t20&locale=zh&luxury_pre_launch=true&map_toggle=true&metadata_only=false&query_understanding_enabled=true&refinement_paths%5B%5D=%2Fhomes&satori_version=1.1.13&screen_size=large&search_by_map=true&selected_tab_id=home_tab&show_groupings=true&supports_for_you_v3=true&sw_lat=28.023354&sw_lng=112.859392&ne_lat=28.025267&ne_lng=112.861948&timezone_offset=480&toddlers=0&version=1.5.7&zoom=19',
    #    headers)

    # request_by_proxy('http://icanhazip.com/',headers)