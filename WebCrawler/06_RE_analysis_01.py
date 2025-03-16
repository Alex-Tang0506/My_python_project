'''Regular Expression'''
'''re package'''

import re

str = '我的电话号是：10086，你的电话号是10010'

# 1. re.finall: 匹配字符串中所有符合的内容
list_1 = re.findall('\d+', str)
print(list_1)

# 2. finditer:匹配字符串中所有的内容[返回的是迭代器]
iter = re.finditer('\d+',str)
print(iter)
for i in iter:
    print(i.group())

# 3. search() 返回 match 中的对象, search是全文匹配并且只要检索到一个就直接返回

s = re.search('\d+',str)
print(s.group())

# 4. match() 从头开始匹配
# m = re.match('\d+',str)
# print(m.group())

#5. 预加载正则表达式
re = re.compile(r"\d+")
a_1 = re.findall(str)
a_2 = re.search(str)
a_3 = re.finditer(str)
print(a_1)
print(a_2)
print(a_3)

