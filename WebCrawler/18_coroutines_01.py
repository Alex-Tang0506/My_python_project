# 导入异步协程包
import asyncio
import time


def func():
    print("This is a thread")


async def func_1():
    print("This is coroutine_1")
    # time.sleep(3)
    await asyncio.sleep(3) # 异步，并挂起
    print("This is coroutine_1 after sleep")


async def func_2():
    print("This is coroutine_2")
    # time.sleep(10)
    await asyncio.sleep(10)
    print("This is coroutine_2 after sleep")


async def func_3():
    print("This is coroutine_3")
    # time.sleep(20)
    await asyncio.sleep(20)
    print("This is coroutine_3 after sleep")


async def func_4():
    print("This is coroutine_4")
    # time.sleep(5)
    await asyncio.sleep(5)
    print("This is coroutine_4 after sleep")


async def func_5():
    print("This is coroutine_5")
    # time.sleep(8)
    await asyncio.sleep(8)
    print("This is coroutine_5 after sleep")


async def func_6():
    print("This is coroutine_6")
    # time.sleep(1)
    await asyncio.sleep(1)
    print("This is coroutine_6 after sleep")


# if __name__ == '__main__':
#     co_routine_1 = func_1()
#     co_routine_2 = func_2()
#     co_routine_3 = func_3()
#     co_routine_4 = func_4()
#     co_routine_5 = func_5()
#     co_routine_6 = func_6()
#
#     # Test_1. 执行同步任务，记录时间
#     # start_rou = time.time()
#     # asyncio.run(co_routine_1)
#     # asyncio.run(co_routine_2)
#     # asyncio.run(co_routine_3)
#     # asyncio.run(co_routine_4)
#     # asyncio.run(co_routine_5)
#     # asyncio.run(co_routine_6)
#     # end_rou = time.time()
#     # print("同步执行时间为:{}".format(end_rou - start_rou))
#
#     # Test_2. 执行异步操作，记录时间
#     # 将任务打包
#     task = [co_routine_1, co_routine_2, co_routine_3, co_routine_4, co_routine_5, co_routine_6]
#
#     # 将打包文件丢入asyncio模块，一次性异步协程启动多个任务,记录时间
#     start_cor = time.time()
#     asyncio.run(asyncio.wait(task))
#     end_cor = time.time()
#     print("异步执行时间为:{}".format(end_cor - start_cor))

# 定义协程方法,该方法用于协程跑以上定义的任务
async def main():
    task = [func_1(), func_2(), func_3(), func_4(), func_5(), func_6()]
    await asyncio.wait(task)

if __name__ == '__main__':
    t_start = time.time()
    asyncio.run(main())
    t_end = time.time()
    print("异步执行时间为:{}".format(t_end - t_start))