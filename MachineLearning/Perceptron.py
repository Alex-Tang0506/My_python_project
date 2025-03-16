import numpy as np
import matplotlib.pyplot as plt


def dic_create(list1):
    dict_new_content = {}
    for i in list1:
        if i not in dict_new_content:
            dict_new_content[i] = 1
        else:
            dict_new_content[i] += 1
    return dict_new_content


def new_feature(list1, list2):
    list3 = [0] * len(list2)
    list4 = []
    for i in list1:
        for j in i:
            list3[list2.index(j)] = 1
        list4.append(list3)
        list3 = [0] * len(list2)
    return list4


def transferlabel(list1):
    list2 = []
    for i in list1:
        if i == '-1':
            list2.append(0)
        else:
            list2.append(1)
    return list2


def dotProduct(list1, list2):
    dotpro_result = 0.0
    for i in range(0, len(list1)):
        dotpro_result += list1[i] * list2[i]
    return dotpro_result


def vectorSubtracion(list1, list2):
    sub_result = [0] * len(list1)
    for i in range(0, len(list1)):
        sub_result[i] = list1[i] - list2[i]
    return sub_result


def vectorAdd(list1, list2):
    Add_result = [0] * len(list1)
    for i in range(0, len(list1)):
        Add_result[i] = list1[i] + list2[i]
    return Add_result


class Perceptron:
    def __init__(self, learning_rate=0.05, learning_loop=5000):
        self.lr_rate = learning_rate  # 初始化学习率
        self.lr_loop = learning_loop  # 初始化学习次数
        self.sig_fun_value = self.sig_fun(1)  # 初始化激活函数值

    '''Define active function'''

    def sig_fun(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    def trainning(self, x):  # 接受数据集，x 为二维数组，自带labels
        _weight = [0] * len(x[0][1:])
        _dot_product = 0.0
        _b = 1
        _w_trained = []
        for _loop in range(self.lr_loop):
            for _i in x:
                _dot_product += dotProduct(_i[1:], _weight)
                _dot_product = _dot_product + _b
                _yPredict = self.sig_fun(_dot_product)
                if _yPredict > 0:
                    _i[1:] = [x * -_yPredict for x in _i[1:]]
                    _w_trained = [x * self.lr_rate for x in _i[1:]]
                    _weight = vectorAdd(_weight, _w_trained)
                    _b = _b - _yPredict * self.lr_rate
                    _dot_product = 0
                else:
                    _dot_product = 0
                    break
        return _weight


f = open(r'C:\Users\23619\Desktop\课程作业和课件\Graduate program\Spring 2021\Machine learning\Mini-project\Final test.txt',
         'r')
ds = f.readlines()
list_labels = []
list_features = []
dict_features = {}
list_all_features = []
list_new_features = []
list_new_labels = []

'''Pre-processing data'''
for i in ds:
    list_features.append(i.split()[1:])
    list_labels.append(i.split()[0])
list_features_1dem = ([i for item in list_features for i in item])

for key in dic_create(list_features_1dem).keys():
    list_all_features.append(key)
list_new_features = new_feature(list_features, list_all_features)

list_new_labels = transferlabel(list_labels)
for i in range(0, len(list_new_features)):
    list_new_features[i].insert(0, list_labels[i])

'''Initialize Perceptron'''
PEC = Perceptron()
list_weight = PEC.trainning(list_new_features)

'''Test'''
x = []
y = []
count_correct = 0
count_incorrect = 0
for i in list_new_features[5:50]:
    value = dotProduct(i[1:], list_weight)
    x.append(i[0])
    if value >= 0:
        y.append('+1')
    else:
        y.append('-1')

for i in range(len(y)):
    if y[i] == x[i]:
        count_correct += 1
    else:
        count_incorrect += 1
print('original test data:', x)
print('predict data:', y)
print('The predict accurate rate = {}'.format(count_correct / (count_correct + count_incorrect)))

plt.figure()
plt.xlabel('Blue point = Original test data')
plt.ylabel('Red point = Prediction error')
x1 = np.array(x)
y1 = np.array(y)
plt.plot(y, 'o', color='r')
plt.plot(x, 'o', color='b')
plt.show()
