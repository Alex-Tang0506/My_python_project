'''Find keyword and the left and right side words'''
import os
import re
import numpy as np
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

path = os.getcwd()

# with open(path + r'\Brother Jacob(1864)_test_Rows.txt', 'r') as f:
#     file_Adam = f.read()
#
# keyword_1 = "Brother"
# print(file_Adam.split())
# print(file_Adam.splitlines())

################################################################################################
################################################################################################
''' Function list to string '''


def list_to_string(list):
    _str1 = ' '.join(list)
    return _str1


''' Function ordered by keyword(Base on the ordered keyword list to change corresponding list)'''


def order_Change(Keyword_Sorted, Index_list, Sub_Array_To_Be_Sorted):

    if len(Keyword_Sorted) != len(Sub_Array_To_Be_Sorted):
        return ["Null"] * len(Keyword_Sorted)

    _sorted_list = Keyword_Sorted
    _index_list = Index_list
    _new_sorted_list = [0] * len(_sorted_list)
    for i in range(0, len(_index_list)):
        _new_sorted_list[i] = Sub_Array_To_Be_Sorted[_index_list[i]]
    return _new_sorted_list


def keyword_l_r(file, File_name, keyword, Left_n, Right_n, fuzzy_sensitive):
    # file must be use "Open" as f and f.read function to get the result with only "r" module
    _file_to_list = file.split()
    # np_file = np.array(_file_to_list)
    _left_n = Left_n
    _right_n = Right_n
    _fuzzSen = fuzzy_sensitive

    '''
    Create 5 list to store 
    # 1.ID 
    # 2.File name
    # 3.leftContext
    # 4.Hit(keyword)
    # 5.RightContext
    '''
    _rows = []
    _leftContext = []
    _rightContext = []
    _Hit = []
    _id = []
    _strCheck = ":"
    # _intCheck = 0
    _fileName = File_name

    for i in range(0, len(_file_to_list)):
        if fuzz.ratio(keyword, _file_to_list[i]) >= _fuzzSen:
            '''1. add hit list'''
            _Hit.append(_file_to_list[i])

            '''2. add _rows list'''
            for _num in range(1, 100):
                _rowsContent = _file_to_list[i - _num]
                if _strCheck in _rowsContent:
                    _rowsContent = _rowsContent.rstrip(_strCheck)
                    if _rowsContent.isdigit() == True:
                        _rowsContent = int(_rowsContent)
                        # if _intCheck < _rowsContent:
                        _rows.append(_rowsContent)
                        # _intCheck = _rowsContent
                        break
                    else:
                        continue
                else:
                    continue

            '''3. add left/right context list'''
            if i - _left_n < 0:
                _leftContext.append(_file_to_list[0: i])
            else:
                _leftContext.append(_file_to_list[i - _left_n: i])

            if i + _right_n > len(_file_to_list):
                _rightContext.append(_file_to_list[i + 1: len(_file_to_list)])
            else:
                _rightContext.append(_file_to_list[i + 1: i + _right_n + 1])
        else:
            continue
    '''4. add id list'''
    for j in range(0, len(_leftContext)):
        _id.append(j + 1)

    '''5. add fileName list'''
    _fileName_list = [_fileName] * len(_leftContext)

    for k in range(0, len(_leftContext)):
        _leftContext[k] = list_to_string(_leftContext[k])
        _rightContext[k] = list_to_string(_rightContext[k])
    _id = list(map(str, _id))
    _rows = list(map(str, _rows))

    '''Reorder all the list base on the Hit'''
    _Hit_sorted = sorted(_Hit)
    _index_list = sorted(range(len(_Hit)), key=_Hit.__getitem__)
    _rows_sorted = order_Change(_Hit_sorted, _index_list, _rows)
    _leftContext_sorted = order_Change(_Hit_sorted, _index_list, _leftContext)
    _rightContext_sorted = order_Change(_Hit_sorted, _index_list, _rightContext)

    return _fileName_list, _rows_sorted, _leftContext_sorted, _Hit_sorted, _rightContext_sorted

def key_word_search(fileInLines, keyword):
    print(len(fileInLines))
    for content in fileInLines:
        if keyword in content:
            return content


# a, b, c, d, e = keyword_l_r(file_Adam, 'Adam Bede(1859)_test.txt', keyword_1, 8, 75)
# # print(a)
# print(b)
# # print(len(c))
# print(d)
# # print(len(e))
