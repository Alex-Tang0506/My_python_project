import numpy as np
from sklearn.datasets import load_svmlight_file
from joblib import Memory
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


mem = Memory("./mycache")
@mem.cache
def get_data():
    data = load_svmlight_file(
        r'C:\Users\23619\Desktop\课程作业和课件\Graduate program\Spring 2021\Machine learning\Mini-project 2\mnist.scale')
    return data[0], data[1]


class supportVectorMachine():
    def __init__(self, learningstep = 2, learningrate = 0.3, penalty_c = 0.5):
        self.lr_step = learningstep
        self.lr_rate = learningrate
        self.c = penalty_c

    def predict(self,w,x,b,raw= False):
        _y = np.dot(w,x) + b
        if raw:
            return _y
        return np.sign(_y)

    def trainningMachine(self, x_input, y_input, target_input):
        _w = np.random.randn(len(x_input[0]))
        _b = 0
        _err = np.zeros(len(y_input))
        for _loop in range(0,self.lr_step):
            _w = _w * self.lr_rate
            for _i1 in range(0,len(y_input)):
                if y_input[_i1] == target_input:
                    continue
                else:
                    _y_predict = self.predict(_w, x_input[_i1], _b,True)
                    e = 1 - _y_predict * y_input[_i1]
                    _err[_i1] = e
                    _idx = np.argmin(_err)
                    if _err[_idx] <= 0:
                        continue
                    else:
                        _delta = self.lr_rate * self.c * y_input[_i1] * x_input[_i1]
                        _w = (1-self.lr_rate) * _w + _delta
                        _b = _b + self.lr_rate * self.c * y_input[_i1]
        return _w, _b

    def test(self,x_input, y_input, weight, bias, target):
        _predictList = np.zeros(len(y_input))
        _originList = np.zeros(len(y_input))
        for _i in range(0,len(x_input)):
            _yPredict = self.predict(weight, x_input[_i], bias,True)
            if _yPredict < 0:
                _predictList[_i] = 1
            else:
                continue
        for _j in range(0,len(_originList)):
            if y_input[_j] != target:
                continue
            else:
                _originList[_j] = 1
        return _predictList, _originList

'''Preprocess data'''
Features, Labels = get_data()
Features = Features.toarray()
Labels = Labels.astype(int)
training_proportion = 0.7

'''Training set'''
label_training = Labels[0:int(len(Labels) * training_proportion)]
feature_training = Features[0:int(len(Features) * training_proportion)]

'''Test set'''
label_test = Labels[int(len(Labels) * training_proportion):]
feature_test = Features[int(len(Labels) * training_proportion):]
print(label_training)
print(len(feature_training[0]))


svm = supportVectorMachine()
w, b = svm.trainningMachine(feature_training, label_training, 4)
predict_label, origin_label = svm.test(feature_test, label_test, w, b, 4)

print(w)
print(b)

'''Test'''
rate_list = np.zeros(len(predict_label))
for idx in range(0, len(predict_label)):
    if predict_label[idx] == origin_label[idx]:
        rate_list[idx] = 1
    else:
        continue
accuracy = np.count_nonzero(rate_list == 1) / len(rate_list)
print('Origin label list = {}'.format(origin_label))
print('Predict label list = {}'.format(predict_label))
print('Accuracy = {}'.format(accuracy))
C_M = confusion_matrix(predict_label, origin_label)
print('Confusion matrix is:\n{}'.format(C_M))

plt.figure()
plt.xlabel('Compare the first 20 result')
plt.ylabel('Value')
pre = predict_label[0:20] + 0.5
ori = origin_label[0:20] + 0.5

x = list(range(len(pre)))
total_width, n = 0.5, 2
width = total_width / n

plt.bar(x, pre, width=width, label='predict', fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, ori, width=width, label='raw',  fc='b')
plt.legend()
plt.show()



