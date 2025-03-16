a_1 = [1,3,6,2,5,4,7]
a_l = [1,3,6]
a_r = [5,4,7]

p = 0
q = len(a_1)
print(p)
print(q)
def MSS_algo(array_A):
    _p = 0
    _q = len(array_A)
    if _p == _q:
        if array_A[_p] > 0:
            return array_A[_p]
        else:
            return 0
        # 数组左右部分相加
        _left_partial_sum = 0
        _right_partial_sum = 0

        # 左右部分最大
        _max_left = 0
        _max_right = 0

        # 左右相加最大
        _left_max_sum = 0
        _right_max_sum = 0

        # 计算中心值
        _center = floor((_p+_q)/2)

        _max_left = MSS_algo(array_A[_p:_center])
        _max_right = MSS_algo(array_A[_center+1:_q])


