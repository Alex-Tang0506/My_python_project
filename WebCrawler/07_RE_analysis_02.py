'''用正则抽取数据'''
import re
str ="""
         https://www.douyin.com/user/MS4wLjABAAAA1IiYWkOlDDCAYN7Mje22I21-BQLBdRLtEs_3g7H7coE?previous_page=app_code_link
     """
# 需求，提取西游记，10010，中国联通
# ?P<xx>从正则匹配的内容中提取想要的内容
re = re.compile(r"https.*?user/(?P<content>.*?)previous.*",re.S) # re.S 让.能匹配换行符
obj = re.finditer(str)
for i in obj:
    # print(i.group('name'))
    print(i.group('content'))