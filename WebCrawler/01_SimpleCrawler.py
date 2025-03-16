from urllib.request import urlopen
url = "http://www.google.com"
resp = urlopen(url)
cont_baidu = resp.read().decode("windows-1252")
print(cont_baidu)
fileName = 'mybaidu.html'

with open(fileName, mode='w',encoding='utf-8') as f:
    f.write(cont_baidu)
print('over')
