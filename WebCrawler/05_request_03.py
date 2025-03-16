import requests

url = "https://movie.douban.com/j/new_search_subjects"
parameters = {
    "sort": "U",
    "range": "0,10",
    "tags":"",
    "start": "0",
    "genres": "喜剧",
}
request_header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
}

content_resp = requests.get(url, params= parameters, headers=request_header)
print(content_resp.url)
print(content_resp.json())
content_resp.close()
