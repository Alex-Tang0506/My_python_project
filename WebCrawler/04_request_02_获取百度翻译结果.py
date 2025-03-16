import requests
url = "https://fanyi.baidu.com/sug"
word = input('Enter a word:')
key_word = {
    "kw": word
}
#Sent post request
cont_resp = requests.post(url ,data=key_word)
print(cont_resp.json()) # 将服务器转换的内容换成jason
cont_resp.close()