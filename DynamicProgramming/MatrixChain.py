'''
Optimize the process of the 3 matrix multiplication that minimize the compute value
a_1: (10*100), a_2=(100*5), a_3=(5*50)
case_1 = ((a_1 * a_2) * (a_3))
T(n) of case_1 = T(a_1 * a_2) + T((a_1 * a_2) * (a_3)) = 10 * 100 * 5 + 10 * 5 * 50 = 5000 + 2500 = 7500

case_2 = ((a_1) * (a_2 * a_3))
T(n) if case_2 = T(a_2 * a_3) + T((a_2 * a_3) + (a_1)) = 100 * 5 * 50 + 10 * 100 * 50 = 25000 + 50000 = 75000
'''


def matrix_chain_order(p):
    n = len(p) - 1
    m = [[0 for j in range(0, n + 1)] for i in range(0, n + 1)]
    s = [[0 for j in range(0, n + 1)] for i in range(0, n + 1)]
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = float("inf")
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k
    return m, s


def print_optimal_parens(s, i, j):
    if i == j:
        print("A", i, end='')
    else:
        print("(", end='')
        print_optimal_parens(s, i, s[i][j])
        print_optimal_parens(s, s[i][j] + 1, j)
        print(")", end='')


if __name__ == '__main__':
    p = [5, 10, 3, 12, 5, 50, 6]
    m, s = matrix_chain_order(p)
    print(m)
    print(s)
    print_optimal_parens(s, 1, 6)
    print('\n')

    for i in range(1, 7):
        for j in range(1, 7):
            print(m[i][j], ' ', end='')
        print()
    print('\n')

    for i in range(1, 7):
        for j in range(1, 7):
            print(s[i][j], ' ', end='')
        print()

