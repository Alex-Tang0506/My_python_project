# 代理是绕过IP封锁的一种手段
import requests

# 寻找免费代理IP
# ip: 52.183.8.192
# port: 3128

proxies = {
    "https": "https://52.183.8.192:3128"
}

main_page = requests.get("https://www.baidu.com", proxies=proxies)
main_page.encoding = "utf-8"
print(main_page.text)