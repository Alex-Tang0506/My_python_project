'''Request 处理防盗链'''
'''防盗链是一种反爬技术，本质是A->B->C->D，如果我从一个页面C开始并对D进行请求，防盗链会检查D的溯源A和B，如果检查不到则不会响应请求'''
'''通过Referer拿到溯源链接()，即 A,B'''
import requests
from fake_useragent import UserAgent
user_agent = UserAgent()

def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent,
        "Referer": "https://www.pearvideo.com/video_1749810"
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp

#
url = 'https://www.pearvideo.com/video_1749810'
videoStatus = 'https://www.pearvideo.com/videoStatus.jsp?contId=1749810&mrd=0.7498647706700219'
# 1.拿到 contId
# 2.拿到 videoStatus 返回的json. -> srcURL
# 3.srcURL里面的内容进行修整
# 4.下载视频
url_list = list(url)
contId_index = url_list.index('_')
contId = url_list[contId_index+1:]
contId = ''.join(contId)
videoStatus_replace = f'https://www.pearvideo.com/videoStatus.jsp?contId={contId}&mrd=0.7498647706700219'

resp = get_source(videoStatus_replace)
print(resp.json())
srcURL = resp.json()['videoInfo']['videos']['srcUrl']
system_time = resp.json()['systemTime']

srcURL = srcURL.replace(system_time, f"cont-{contId}")
print(srcURL)
