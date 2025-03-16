import os
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

path = os.getcwd()

with open(path + r'\Adam Bede(1859)_test_Rows.txt', 'r') as f:
    file_Adam = f.read()
    file_Adma_lines = file_Adam.splitlines()

keyword_1 = "20665:"
fuz_accurcy = 75
test_1 = "123.."
test_2 = "b"
test_3 = "c"
str_1 = '.'
print(file_Adma_lines)


def keyWord(fileInLines, keyword):
    print(len(fileInLines))
    for content in fileInLines:
        if keyword in content:
            return content

a = keyWord(file_Adma_lines, keyword_1)
print(a)














#
# list_1 = [1,3,57,9,2,4,0,8,7]
# index = sorted(range(len(list_1)), key=list_1.__getitem__)
# print(index)
#
#
# def order_Change(Keyword_Sorted, Index_list, Sub_Array_To_Be_Sorted):
#     if len(Keyword_Sorted) != len(Sub_Array_To_Be_Sorted):
#         return None
#
#     _sorted_list = Keyword_Sorted
#     _index_list = Index_list
#     _new_sorted_list = [0] * len(_sorted_list)
#     for i in range(0, len(_index_list)):
#         _new_sorted_list[i] = Sub_Array_To_Be_Sorted[_index_list[i]]
#     return _new_sorted_list
#
# keyword_sorted = sorted(keyword_1)
# index_list = sorted(range(len(keyword_1)), key=keyword_1.__getitem__)
# print(keyword_1)
# print(index_list)
# a = order_Change(keyword_sorted, index_list, rowList_1)
# print(a)
# b = ["Null"] * 5
# print(b)
