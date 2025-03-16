# 进程是资源单位，线程是执行单位，每一个进程至少要有一个线程
# 启动程序默认都有一个主线程
#
# 单线程的例子
# def fun():
#     for i in range(1000):
#         print(i)
#
# if __name__ == '__main__':
#     fun()
#     for i_1 in range(1000):
#         print(i_1)

# 多线程
from threading import Thread
from multiprocessing import Process


#
#
def func(name):
    for i_1 in range(1000):
        print(name, i_1)


#
#
# if __name__ == "__main__":
#     # 初始化线程,并设置对象为func,即，变量t是用来执行func的功能，并同时执行main中的功能
#     t_1 = Thread(target=func)
#     t_2 = Thread(target=func)
#     t_1.start()  # 多线程状态为可以开始工作状态，具体执行时间由CPU决定
#     t_2.start()
#
#     for i_2 in range(1000):
#         print("main", i_2)

# t_1 = Thread(target=func, args=('线程1',))
# t_1.start()
#
# t_2 = Thread(target=func, args=('线程2',))
# t_2.start()

# 线程池和进程池
# 一次性开辟N个线程，用户直接给线程池提交任务。具体执行CPU决定

'''step.1 导入线程池和进程池的包'''
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor

if __name__ == "__main__":
    # 创建线程池 (50个线程)
    with ThreadPoolExecutor(50) as t:
        for i in range(100):
            t.submit(func, name=f'线程{i}')
    print("123")