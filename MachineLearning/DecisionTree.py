import math

def entropycompute(list):
    _numSample = len(list)
    _dict = {}
    _entropy = 0.0
    for _label in list:
        if _label not in _dict.keys():
            _dict[_label] = 0
        _dict[_label] += 1
    for key in _dict:
        _prob = float(_dict[key]) / _numSample
        _entropy -= _prob * math.log(_prob, 10)
    return _entropy

f = open(r'C:\Users\23619\Desktop\课程作业和课件\Graduate program\Spring 2021\Machine learning\Mini-project\Data set a4a(Binary classification).txt','r')
listds = []        #所有数据
listds1 = []       #+1 属性数据
listds0 = []       #-1 属性数据
listla = []        #标签
listOut = []       #记录相同属性并全部列出
list_entropy = []  #存储熵
list_infogain = [] #存储 information gain
tree_Node = []     #存储节点
root_list = []
'''----------------------------------- Preprocess dataset -----------------------------------'''
ds = f.readlines()
for line in ds:
    listds.append(line.split())

for i in listds:
    for j in i:
        if j == '-1':
            listds0.append(i[1:])
            listla.append(j)
        elif j == '+1':
            listds1.append(i[1:])
            listla.append(j)
        else:
            continue

list_ds1_1dimen = ([i for item in listds1 for i in item])
list_ds0_1dimen = ([i for item in listds0 for i in item])
list_all_1dimen = ([i for item in listds1 + listds0 for i in item])

'''----------------------------------- 1. Root node sorted -----------------------------------'''
# root sorted
for i in list_ds0_1dimen:
    for j in range(0,len(list_ds1_1dimen)):
        if i == list_ds1_1dimen[j]:
            listOut.append(i+'-1')
        else:
            continue
    for j1 in range(0,len(list_ds0_1dimen)):
        if i == list_ds0_1dimen[j1]:
            listOut.append(i+'-0')
        else:
            continue
    list_entropy.append('%s=%f'%(i,entropycompute(listOut)))
    listOut.clear()

#compute information gain
for i in list_entropy:
    s1 = i.split('=')
    list_infogain.append([s1[0], entropycompute(listla) - float(s1[1]) * (list_ds1_1dimen.count(s1[0]) / len(listds) + list_ds0_1dimen.count(s1[0]) / len(listds))])
node = sorted(dict(list_infogain).items(), key=lambda item:item[1], reverse=True)
root = list(node[0])

'''----------------------------------- 2.Child node sorted -----------------------------------'''
_listds= []
_listds0 = []
_listds1 = []
_listla = []
_listOut = []
_list_entropy = []  #存储熵
_list_infogain = [] #存储 information gain
_listnode = []     #存储节点

for _line in ds:
    _listds.append(_line.split())
for _content in _listds:
    if root[0] in _content:
        _listds1.append(_content[_content.index(root[0]):])
        _listla.append(root[0])
    else:
        _listds0.append(_content[1:])
        _listla.append(root[0] + '0')

_list_ds1_1dimen = ([i for item in _listds1 for i in item])
_list_ds0_1dimen = ([i for item in _listds0 for i in item])
list_all_1dimen = ([i for item in _listds1 + _listds0 for i in item])

for _i in _list_ds0_1dimen:
    for j in range(0,len(_list_ds1_1dimen)):
        if _i == _list_ds1_1dimen[j]:
            _listOut.append(_i+'-1')
        else:
            continue
    for _j1 in range(0,len(_list_ds0_1dimen)):
        if _i == _list_ds0_1dimen[_j1]:
            _listOut.append(_i+'-0')
        else:
            continue
    _list_entropy.append('%s=%f' % (_i, entropycompute(_listOut)))
    _listOut.clear()
for _i in _list_entropy:
    _s1 = _i.split('=')
    _list_infogain.append([_s1[0], entropycompute(_listla) - float(_s1[1]) * (_list_ds1_1dimen.count(_s1[0]) / len(_listds) + _list_ds0_1dimen.count(_s1[0]) / len(_listds))])
    _listnode = sorted(dict(_list_infogain).items(), key=lambda item:item[1], reverse=True)

list.insert(_listnode,0,tuple(root))
list_root_key = list(dict(_listnode).keys())
print(list_root_key)

'''----------------------------------- Test -----------------------------------'''
count_correct = 0
count_wrong = 0
for i in listds:
    for j in i[1:]:
        if j in list_root_key:
            if i[0] == '-1':
                count_correct+=1
            else:
                count_wrong+=1
        else:
            count_wrong+=1
print('accuracy = {}'.format(count_correct/(count_wrong+count_correct)))


