#coding:utf-8
# Python 实现冒泡排序
def bubbleSort(alist):
    for passnum in range(len(alist)-1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i+1]:
                alist[i], alist[i+1] = alist[i+1], alist[i]
    return alist

alist = [54,26,93,17,77,31,44,55,20]
#print(bubbleSort(alist))

# 改进的冒泡排序, 加入一个校验, 如果某次循环发现没有发生数值交换, 直接跳出循环
def modiBubbleSort(alist):
    exchange = True
    passnum = len(alist) - 1
    while passnum >= 1 and exchange:
        exchange = False
        for i in range(passnum):
            if alist[i] > alist[i+1]:
                alist[i], alist[i+1] = alist[i+1], alist[i]
                exchange = True
        passnum -= 1
    return alist

#print(bubbleSort(alist))


def midbub(alist):
    ex = True
    passnum = len(alist)-1
    while passnum>=1 and ex:
        ex = False
        for i in range (0,passnum):
            if alist[i] > alist[i+1]:
                alist[i],alist[i+1] = alist[i+1],alist[i]
                ex = True
        passnum -=1
    return alist
#print (midbub(alist))

def bub_rec(alist,n):
    if n ==1:
        return
    for i in range(n):
        if alist[i]>alist[i+1]:
            alist[i],alist[i+1] = alist[i+1],alist[i]
    bub_rec(alist,n-1)

print bub_rec(alist,len(alist)-1)
print alist