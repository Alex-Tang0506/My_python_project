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

def new_feature(list1,list2):
    list3 = [0] * len(list2)
    list4 = []
    for i in list1:
        for j in i:
            list3[list2.index(j)] = 1
        list4.append(list3)
        list3 = [0] * len(list2)
    return list4

class KNN:
    def __init__(self):
        self.k = 20           #Initialize K value

    # Euclidean distance calculator
    def euclideanDistance(self,list1, list2):
        _distance = 0
        length = len(list2)
        for _x in range(length):
            _distance += np.square(list1[_x] - list2[_x])
        return np.sqrt(_distance)

    def training(self,training_list,test_list):
        _distance = []
        _length = len(test_list)
        _vote = {}

        #training machine, sorted the nearest distance, return a list from small to large
        for x in range(len(training_list)):
            _disValue = self.euclideanDistance(training_list[x][1:], test_list[1:])
            _distance.append([training_list[x][0], _disValue])
        _sorted_distance = sorted(_distance,key=lambda x: x[1])

        #vote machine, select K nearest point,count the labels and vote, format = (result + vote number)
        for i in range(self.k):
            if _sorted_distance[i][0] not in _vote:
                _vote[_sorted_distance[i][0]] = 1
            else:
                _vote[_sorted_distance[i][0]] += 1
        result = sorted(_vote.items(), key=lambda x: x[1], reverse=True)
        return result[0]

f = open(r'C:\Users\23619\Desktop\课程作业和课件\Graduate program\Spring 2021\Machine learning\Mini-project\Final test.txt','r')
ds = f.readlines()
list_labels = []                             #原始标签
list_features = []                           #原始属性
dict_features = {}                           #字典格式属性
list_all_features = []                       #所有属性，1维列表
list_new_features = []                       #标签+属性01集合
count_correct = 0                            #统计预测正确的数字
count_incorrect = 0                          #统计预测不正确的数字
Knn = KNN()                                  #Initialize Knn
'''读取原始ds文件，分别分出标签组和属性组'''
for i in ds:
    list_features.append(i.split()[1:])
    list_labels.append(i.split()[0])
list_features_1dem = ([i for item in list_features for i in item])

'''生成不重复的新属性组，即总共有多少种属性，将总数和原始数据属性合并，得到10表示的所有原始属性'''
for key in dic_create(list_features_1dem).keys():
    list_all_features.append(key)
list_new_features = new_feature(list_features,list_all_features)

'''将得到的10表示的新属性和原始标签合并，生成新的数据集合['-1',0,0,1],['1',1,0,1] 的格式'''
for i in range(0,len(list_new_features)):
    list_new_features[i].insert(0,list_labels[i])

# print(list_labels)
# print(list_features)
# print(list_all_features)
print(list_new_features)

'''Test data, select rows 5-30 to test the result'''
x = []  #original data
y = []  #predict data
for i in list_new_features[5:30]:
    predict_result = Knn.training(list_new_features,i)
    y.append(predict_result[0])
    x.append(i[0])
    print(predict_result,i[0])
    if predict_result[0] == i[0]:
        count_correct +=1
    else:
        count_incorrect += 1

print('original test data:',x)
print('predict data:',y)
print('The predict accurate rate = {}'.format(count_correct/(count_correct + count_incorrect)))
plt.figure()
plt.xlabel('Blue point = Original test data')
plt.ylabel('Red point = Prediction error')
x1 = np.array(x)
y1 = np.array(y)
plt.plot(y,'o',color = 'r')
plt.plot(x,'o',color = 'b')
plt.show()






