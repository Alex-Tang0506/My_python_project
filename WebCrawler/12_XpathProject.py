import requests
import fake_useragent
from lxml import etree

user_agent = fake_useragent.UserAgent()


def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp

# 1.拿页面源代码
url = 'https://www.zbj.com/search/f/?kw=saas'
code_Mainpage = get_source(url)
html = etree.HTML(code_Mainpage.text)  # 用html 方法加载原始网页

# 读取价格，企业名，所在城市，项目名

all_area = html.xpath('/html/body/div[6]/div/div/div[2]/div[5]/div[1]/div')
for sub_area in all_area:
    company_Name = sub_area.xpath('./div/div/div[2]/div/text()')
    price = sub_area.xpath('.')
    print(company_Name)