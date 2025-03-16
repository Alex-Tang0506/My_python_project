'''爬电影天堂数据，dytt89.com'''
# Task 1：拿到2021必看热片部分的所有下载链接
# step_1: 从首页定位到2021必看热片
# step_2: 从2021必看热片定位到下载
# step_3: 拿到想要的下载地址

import requests
import re
import csv

'''step.1 请求网站源码 '''
url = 'https://dytt89.com/'
# 获取user-agent
request_headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
}

url_resp = requests.get(url, headers=request_headers)

# print(url_resp.encoding)  # 查看request解析出来的编码
# print(url_resp.apparent_encoding)  # 查看页面原编码

url_resp.encoding = 'GB2312'  # 将request解析编码替换为页面原编码，解决解析后的乱码问题
html_code = url_resp.text
# print(html_code)

'''step.2 定位到2021必看热片 从2021必看片中提取子页面的地址，即提取子页面中 href=xxx的信息'''
obj_2021 = re.compile(r"2021必看热片.*?<ul>(?P<name_2021>.*?)</ul>", re.S)
obj_2021_subpage = re.compile(r"<a href='(?P<href>.*?)'.*?>",re.S)
obj_2021_subpage_link1 = re.compile(r'◎片　　名　(?P<moive_name>.*?)<br />.*?<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<link>.*?)"',re.S)
obj_2021_subpage_link2 = re.compile(r'【下载地址】.*?<a href="(?P<download_link>.*?).*?">',re.S)
obj_2021_subpage_link3 = re.compile(r'【下载地址】.*?<a href="(?P<download_link>.*?).*?">',re.S)

result_2021 = obj_2021.finditer(html_code)
child_list_href = []
for i in result_2021:
    ul = i.group('name_2021').strip()
    print(ul)
    result_2021_sub = obj_2021_subpage.finditer(ul)
    for j in result_2021_sub:
        print(j.group('href'))
        sub_page = url + j.group('href').strip('/')
        # print(sub_page)
        child_list_href.append(sub_page)
# print(child_list_href)

'''从子页面中获取内容'''
for href in child_list_href:
    child_page = requests.get(href, headers=request_headers)
    child_page.encoding = 'GB2312'
    html_child_code = child_page.text
    link = obj_2021_subpage_link1.finditer(html_child_code)
    # for i_1 in link:
    #     print(i_1.group('moive_name'))
    #     print(i_1.group('link'))
    # # for i_1 in link:
    # #     print(i_1.group('download_link'))




