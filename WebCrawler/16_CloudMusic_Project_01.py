import fake_useragent
import requests
import time

user_agent = fake_useragent.UserAgent()


def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.post(_url, headers=_request_headers)
    return _url_resp


url = 'https://music.163.com/weapi/comment/resource/comments/get?csrf_token='
main_page = get_source(url)
print(main_page.text)

#1. 找到未加密的参数
#2. 想办法把参数进行加密
#3. params and encSecKey
#4. 请求网易，拿到评论
