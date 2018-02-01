# -*- coding:UTF-8 -*-
import functools
# Python3.x和Python2.x对于map、reduce、filter的处理变得不同
# Python3.x中map和filter的输出是一个map型和filter型, 需要从里面取出需要的值
# Python2.x中map和filter输出的直接是一个list
# Python3.x中使用reduce需要引入functools

# map是用同样方法把所有数据都改成别的..字面意思是映射..比如把列表的每个数都换成其平方..
# reduce是用某种方法依次把所有数据丢进去最后得到一个结果..字面意思是化简..比如计算一个列表所有数的和的过程,就是维持一个部分和然后依次把每个数加进去..
# filter是筛选出其中满足某个条件的那些数据..字面意思是过滤..比如挑出列表中所有奇数..
# >>> map(lambda x:x*x,[0,1,2,3,4,5,6])
# [0, 1, 4, 9, 16, 25, 36]
# >>> reduce(lambda x,y:x+y,[0,1,2,3,4,5,6])
# 21
# >>> filter(lambda x:x&1,[0,1,2,3,4,5,6])
# [1, 3, 5]
# 原先有多少map完后还是有多少..原先不管有多少reduce后都只剩一个结果..filter完则是原先的一部分,也许全都还在,也许全都没了,反正个数不定..
# 但是剩下的那些也都是原先有的..


# 使用map把list中的int变为str
list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
map(str, list)
print list
# 使用map()把名字规范化, 首字母大写,其余小写
def standardName(s):
    return s.capitalize()
print([x for x in map(standardName, ['adam', 'LISA', 'barT'])])
# 在Python2.x中应该使用print(map(standardName, ['adam', 'LISA', 'barT']))

# 使用reduce()输出一个list的所有数的乘积
def prod(aList):
    return functools.reduce(lambda x, y: x*y, aList)
print(prod([1, 2, 3, 4, 5]))
print reduce(lambda x,y:x*y,[1, 2, 3, 4, 5])
# 使用filter()打印100以内的素数
def isPrime(n):
    isPrimeFlag = True
    if n <= 1:
        isPrimeFlag = False
    i = 2

    while i * i <= n:
        if n % i == 0:
            isPrimeFlag = False
            break
        i += 1
    return n if isPrimeFlag else None
print(filter(isPrime, range(101)))

def prime(num):
    for i in range(2, num):
        if num % i == 0:    # 能被1之外的任意个数整除的即为非素数，返回False，将被filter函数过滤掉
            return False
    return True

print'prime： ', filter(prime, range(2, 101))      # filter(func,seq)返回seq作用于func之后为True的


