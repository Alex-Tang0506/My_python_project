import fake_useragent
import requests

# 1. 生成随机user_agent
user_agent = fake_useragent.UserAgent()
def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp

# 2. 下载图片到本地

user_agent = fake_useragent()
headers = {'User-Agent': user_agent.random}
def getImg(img_url, img_Name):
    jpg_url = img_url
    r = requests.get(jpg_url, headers=headers)
    if r.status_code == '200':
        print(img_url + 'success')
    content = r.content
    with open(img_Name, 'wb') as fp:
        fp.write(content)