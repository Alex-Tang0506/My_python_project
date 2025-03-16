'''Given a rod of length n and a list of rod prices of length i, where 1 <= i <= n, find the optimal way to cut the rod into smaller rods to maximize profit.'''
'''
length[] = [1, 2, 3, 4, 5, 6, 7, 8]
price[] = [1, 5, 8, 9, 10, 17, 17, 20]
Rod length: 4
Best: Cut the rod into two pieces of length 2 each to gain revenue of 5 + 5 = 10
'''


def rodCut(price, n):
    # base case
    if n == 0:
        return 0
    maxValue = 0
    # one by one, partition the given rod of length `n` into two parts of length
    # (1, n-1), (2, n-2), (3, n-3), … ,(n-1, 1), (n, 0) and take maximum

    for i in range(1, n + 1):
        cost = price[i - 1] + rodCut(price, n - i)
        if cost > maxValue:
            maxValue = cost
    return maxValue


if __name__ == '__main__':
    price = [1, 5, 8, 9, 10, 17, 17, 20]
    rod_length = 4
    print('Value is:{}'.format(rodCut(price, rod_length)))
