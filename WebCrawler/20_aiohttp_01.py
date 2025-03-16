# 异步请求处理模块
import aiohttp
import asyncio
import fake_useragent
import requests

user_agent = fake_useragent.UserAgent()

urls = [
    "http://kr.shanghai-jiuxin.com/file/2020/1031/d7de3f9faf1e0ecdea27b73139fc8d3a.jpg",
    "http://kr.shanghai-jiuxin.com/file/2020/1031/191468637cab2f0206f7d1d9b175ac81.jpg",
    "http://kr.shanghai-jiuxin.com/file/2020/1031/774218be86d832f359637ab120eba52d.jpg"
]


def get_source(url):
    _url = url
    _user_agent = user_agent.random
    _request_headers = {
        "user-agent": _user_agent
    }
    _url_resp = requests.get(_url, headers=_request_headers)
    return _url_resp


async def aiodownload(url):
    # aiorequest = aiohttp.ClientSession()  # 等价于requests模块
    # aiorequest.get(url)
    file_name = url.split("/", 6)[6]
    async with aiohttp.ClientSession() as aio_Re:
        ''' 
        aio_Re.request()
        aio_Re.get()
        aio_Re.post()
        '''
        async with aio_Re.get(url) as resp:
            with open(file_name, mode="wb") as f:
                f.write(await resp.content.read()) # 读取内容为异步，需要挂起
    print("Task Finish")


async def main():
    _task = []
    for _item in urls:
        _source = aiodownload(_item)
        _task.append(_source)
    await asyncio.wait(_task)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
