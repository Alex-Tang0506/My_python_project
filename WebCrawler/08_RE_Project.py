# 提取豆瓣top250

# step_1:拿到页面源代码  request
# step_2:通过正则表达式提取信息  re模块

import requests
import re
import csv

url = "https://movie.douban.com/top250"
request_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
}
url_resp = requests.get(url, headers=request_header)
page_content = url_resp.text
print(page_content)

# 解析数据, 取得影片名，年份，导演，评分
obj = re.compile(r'<li>.*?<div class="item">.*?<span class="title">(?P<name>.*?)</span>.*?<p class="">(?P<director>.*?)&nbsp;&nbsp;&nbsp;(?P<protagonist>.*?)<br>'
                 r'(?P<year>.*?)&nbsp.*?<div class="star">.*?<span class="rating_num" property="v:average">(?P<rating_num>.*?)</span>', re.S)
result_obj = obj.finditer(page_content)

# 写入csv
f = open("08_re_Project.csv", mode="w", encoding='utf-8-sig')
csv_writer = csv.writer(f)
for i in result_obj:
    print(i.group('name'))
    print(i.group('director').strip())
    print(i.group('protagonist'))
    print(i.group('rating_num'))
    print(i.group('year').strip())
    dic = i.groupdict()
    dic['director'] = dic['director'].strip()
    dic['year'] = dic['year'].strip()
    csv_writer.writerow(dic.values())
f.close()
print('over!')

