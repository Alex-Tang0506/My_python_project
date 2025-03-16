import numpy as np
from sklearn.datasets import load_svmlight_file
from joblib import Memory

"""Data Preprocess"""
mem = Memory("./mycache")


@mem.cache
def get_data():
    data = load_svmlight_file(
        r'mnist.scale')
    return data[0], data[1]


'''
Features: All pixels of 1 sample, number of pixels = 780
Labels: The actual number of 1 sample
length of Features  = length of Labels = 60000
Format : Features [0,0,.....1,5](Total number = 780, or length of this array = 780) -> Labels 5
'''
Features, Labels = get_data()
Features = Features.toarray()
''' 60000 * 780 Matrix '''
print(len(Features[0]))
''' 60000 * 1 Matrix '''
print(Labels)

'''Split data into subset of training and testing'''


def train_test_split(features, lables, train_rate):
    _train_feat = features[:round(len(features) * train_rate)]
    _train_lab = lables[:round(len(lables) * train_rate)]
    _test_feat = features[round(len(features) * train_rate):]
    _test_lab = lables[round(len(lables) * train_rate):]

    return _train_feat, _train_lab, _test_feat, _test_lab


train_data_feat, train_data_lab, test_data_feat, test_data_lab = train_test_split(Features, Labels, 0.8)

'''Define sigmoid function'''


def sigmoid_fun(x):
    _x = 1 / (1 + np.exp(-x))
    return _x


'''Define derivative'''


def deriv_sigmoid(x):
    f_x = x
    return f_x * (1 - f_x)


'''Define loss function'''


def mse_loss(y_true, y_predict):
    return ((y_true - y_predict) ** 2).mean()


class NeuralNetwork:
    """Initialize the parameter"""

    def __init__(self, layer1_node=3, layer2_node=2):
        np.random.seed(1)
        self.l1_node = layer1_node
        self.l2_node = layer2_node
        self.l1_w = np.random.rand(len(train_data_feat[0]), self.l1_node)  # 780 * 3
        self.l1_b = np.random.rand(self.l1_node, 1)  # 3 * 1
        self.l2_w = np.random.rand(self.l1_node, self.l2_node)  # 3 * 2
        self.l2_b = np.random.rand(self.l2_node, 1)  # 2 * 1
        self.l3_w = np.random.rand(self.l2_node, 1)  # 2 * 1
        self.l3_b = np.random.rand(1, 1)  # 1 * 1

    '''Define feedforward'''
    '''x = 780 * 1 matrix'''

    def feedforward(self, x, w_1, b_1, w_2, b_2, w_3, b_3):
        '''Generate layer 1: Neuron h1 '''
        _h_1_sum = x.dot(w_1).reshape(3, 1) + b_1
        _h_1 = sigmoid_fun(_h_1_sum)

        '''Generate layer 2: Neuron h2 '''
        _h_2_sum = (_h_1.T.dot(w_2)).T + b_2
        _h_2 = sigmoid_fun(_h_2_sum)

        '''Generate layer 3: Neuron h3 = output = y_predict '''
        _h_3_sum = _h_2.T.dot(w_3) + b_3
        _h_3 = sigmoid_fun(_h_3_sum)
        _ypred = _h_3

        return _h_1, _h_2, _ypred

    '''Define backpropagation'''

    def backpropagation(self, x, y_true, y_pred, h_1, h_2):
        ''' Calculate partial derivative'''

        '''Partial derivative to l3_w and l3_b'''
        _d_l_d_ypred = -2 * (y_true - y_pred)
        _d_l_d_w3 = _d_l_d_ypred * h_2 * deriv_sigmoid(y_pred)  # 2*1
        _d_l_d_b3 = _d_l_d_ypred * deriv_sigmoid(y_pred)  # 1*1
        #
        '''Partial derivative to l2_w and l2_b'''
        _d_l_d_w2 = (_d_l_d_ypred * deriv_sigmoid(y_pred) * self.l3_w * deriv_sigmoid(h_2) * h_1.T).T  # 3*2
        _d_l_d_b2 = _d_l_d_ypred * deriv_sigmoid(y_pred) * self.l3_w * deriv_sigmoid(h_2)  # 2*1
        #
        '''Partial derivative to l1_w and l1_b'''
        _d_l_d_w1 = ((_d_l_d_ypred * deriv_sigmoid(y_pred) * self.l3_w * deriv_sigmoid(h_2)).T.dot(
            self.l2_w.T) * deriv_sigmoid(h_1).T).T * x
        _d_l_d_b1 = ((_d_l_d_ypred * deriv_sigmoid(y_pred) * self.l3_w * deriv_sigmoid(h_2)).T.dot(
            self.l2_w.T) * deriv_sigmoid(h_1).T).T

        return _d_l_d_w1, _d_l_d_b1, _d_l_d_w2, _d_l_d_b2, _d_l_d_w3, _d_l_d_b3, _d_l_d_ypred

    def train(self, x, y_ture, w_1, b_1, w_2, b_2, w_3, b_3, rate=0.1, loop=100):
        self._rate = rate
        self._loop = loop
        for _i in range(0, len(x)):
            _h1, _h2, _ypred = self.feedforward(x[_i], w_1, b_1, w_2, b_2, w_3, b_3)
            _l_d_w1, _l_d_b1, _l_d_w2, _l_d_b2, _l_d_w3, _l_d_b3, _l_d_ypred = self.backpropagation(x[_i], y_ture[_i],
                                                                                                    _ypred, _h1, _h2)

            '''Update each hidden layer'''

            '''Hidden layer_1'''
            self.l1_w -= _l_d_w1.reshape(780, 3) * self._rate
            self.l1_b -= _l_d_b1 * self._rate

            '''Hidden layer_2'''
            self.l2_w -= _l_d_w2 * self._rate
            self.l2_b -= _l_d_b2 * self._rate

            '''Hidden layer_3'''
            self.l3_w -= _l_d_w3 * self._rate
            self.l3_b -= _l_d_b3 * self._rate

        return self.l1_w, self.l1_b, self.l2_w, self.l2_b, self.l3_w, self.l3_b


NN = NeuralNetwork()

'''Change format'''
train_data_lab = train_data_lab.reshape(len(train_data_lab), 1)
print(train_data_feat)
print(train_data_lab)

w1, b1, w2, b2, w3, b3 = NN.train(train_data_feat, train_data_lab, NN.l1_w, NN.l1_b, NN.l2_w, NN.l2_b, NN.l3_w, NN.l3_b)

h1, h2, y_predict = NN.feedforward(train_data_feat[0], w1, b1, w2, b2, w3, b3)
print(y_predict)
loss = mse_loss(5, y_predict)
print(loss)

