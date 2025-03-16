import requests
query = input('输入一个明星：')
url = f"https://www.sogou.com/web?query={query}"
requ_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
}
'''浏览器中所有的输入搜索方式均为get'''
'step.1 获取响应, 处理反爬（增加headers让服务器认为该程序是通过浏览器访问而非第三方）'
cont_resp = requests.get(url, headers=requ_headers)

print(cont_resp)  # 相应状态为200 = ok 即当前请求没有问题
print(cont_resp.request) #响应方式 = get
print(cont_resp.url) # 响应地址
print(cont_resp.text) # 地址网站源代码
cont_resp.close()


