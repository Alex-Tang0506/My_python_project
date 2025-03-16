# 模拟爬虫中的异步请求
import time
import asyncio


# 定义下载函数
async def download(url):
    print("准备开始下载源码...")
    await asyncio.sleep(5)  # 模拟 request 命令请求的时间
    print("下载完成")


# 定义异步操作函数
async def main():
    _task = []
    _urls = [
        "https://www.baidu.com",
        "https://www.163.com",
        "https://www.bilibili.com"
    ]

    for item in _urls:
        _main_page = download(item)
        _task.append(_main_page)

    await asyncio.wait(_task)


# 主函数中跑通异步函数
if __name__ == '__main__':
    star_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print("下载时间为:{}".format(end_time - star_time))
