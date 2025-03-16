# bs4 是通过标签来定位到数据的一种解决方案
import requests
import fake_useragent
from bs4 import BeautifulSoup

user_agent = fake_useragent.UserAgent()
#生成随机user agent
def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp


url = 'https://apps.lib.ua.edu/blogs/this-goodly-land/author/?AuthorID=11'
main_page = get_source(url)
main_page.encoding = 'utf-8'
main_page_txt = main_page.text
# print(main_page_txt)

# 1. 把页面源代码交给bs4
page = BeautifulSoup(main_page_txt, 'html.parser')  #指定为html解析器
# print(page)

# 2. 从beautifulsoup中查找数据， find（找第一个x）  find_all(找所有x)
# 语法 find(标签，属性)， find_all(标签，属性)
table = page.find_all("h3", class_="author-h3")
print(table)
