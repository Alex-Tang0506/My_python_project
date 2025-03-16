#cookie 的目的
# 在爬数据的时候，很多内容是需要通过用户的登录才能获取到的，比如想查看自己的购物车信息。所以需要获取cookie
'''
所谓cookie，是指用户在登录的时候会向服务器发从请求，这个请求中带着用户的username 和 password，当服务器回应对该请求时，就会发送一串代码，
这个代码就是cookie,保存了username password等用户的信息，该信息会被记录在用户自己的浏览器，在用户下次再次发送登录请求时，就会以cookie的形式向服务器发送。
'''

# 请求cookie的步骤: 1. 登录，获取服务器返回的 cookie  2. 带着cookie去请求书架url->获取书架上的内容
# 可以使用session进行请求，该请求可以理解为上面2步的结合体，并且cookie不会丢失
import requests
session = requests.session()
# session.get()
# session.post()

url = 'https://bbs.nga.cn/nuke.php?__lib=login&__act=account&login'
data = {
    'uid': '63892196',
    'cid': 'X98uoujbbip7cj07t2bbtma1ubif8omoucbrg67d'
}
resp = session.get(url, data=data)
print(resp.text)

session.get()

