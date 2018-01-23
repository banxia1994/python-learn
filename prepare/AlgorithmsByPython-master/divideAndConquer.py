#coding:utf-8
'''
分治算法一般都伴随着递归算法
'''
# 分治算法实现查找数组中的最大元素的位置
def maxIndex(alist, start, end):
    if start > end or len(alist) == 0:
        return
    pivot = (start+end) >> 1
    if end - start == 1:
        return start
    else:
        temp1 = maxIndex(alist, start, pivot)
        temp2 = maxIndex(alist, pivot, end)
        if alist[temp1] < alist[temp2]:
            return temp2
        else:
            return temp1
print(maxIndex([5,7,9,3,4,8,6,2,0,1], 0, 9))

# 分治法计算正整数幂
def power(base, x):
    if x == 1:
        return base
    else:
        if x & 1 == 1:
            return power(base, x>>1)*power(base, x>>1)*base
        else:
            return power(base, x>>1)*power(base, x>>1)

print(power(2, 6))

def mIndex(A,start,end):
    piv = (start+end)//2
    if len(A)==1 or start>end:
        return -1
    if end-start ==1:
        return start

    else:
        t1 = mIndex(A,start,piv)
        t2 = mIndex(A,piv,end)
        if A[t1]>=A[t2]:
            return t1
        else:
            return t2
#print(mIndex([5,7,9,3,4,8,6,2,0,1], 0, 9))


#查找第k小的元素，要求线性时间
def par(seq):
    piv = seq[0]  #选择主元 分序列
    low = [x for x in seq[1:] if x<=piv]
    high = [x for x in seq[1:] if x>piv]
    return piv,low,high

def select(seq,k):
    piv,low,high = par(seq)
    m = len(low)
    if len(low)==k-1:
        return piv
    elif len(low)<k-1:
        return select(high,k-m-1)
    else :
        return select(low,k)

seq = [3, 4, 1, 6, 3, 7, 9, 13, 93, 0, 100, 1, 2, 2, 3, 3, 2]
print(select(seq, 1)) #2
print(select(seq, 4)) #2

#快速排序
def quicksort(seq):
    if len(seq)<=1:
        return seq
    piv,low,high = par(seq)

    return quicksort(low)+[piv]+quicksort(high)

seq = [7, 5, 0, 6, 3, 4, 1, 9, 8, 2]
print(quicksort(seq)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#二分排序
def mergesort(seq):
    mid = len(seq)//2
    ls,rs = seq[:mid],seq[mid:]

    if len(ls)>1:
        ls = mergesort(ls)
    if len(rs)>1:
        rs = mergesort(rs)

    res = []
    while ls and rs:
        if ls[-1] >= rs[-1]:
            res.append(ls.pop())
        else:
            res.append(rs.pop())

    res.reverse()
    return (ls or rs) + res
seq = [7, 5, 0, 6, 3, 4, 1, 9, 8, 2]
print(mergesort(seq)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#找出和位s 的所有组合
def find(seq, s):
    n = len(seq)
    if n == 1:
        return [0, 1][seq[0] == s]

    if seq[0] == s:
        return 1 + find(seq[1:], s)
    else:
        return find(seq[1:], s - seq[0]) + find(seq[1:], s)


# 测试
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
s = 14  # 和
print(find(seq, s))  # 15

seq = [11, 23, 6, 31, 8, 9, 15, 20, 24, 14]
s = 40  # 和
print(find(seq, s))  # 8


def find2(seq, s, tmp=''):
    if len(seq) == 0:  # 终止条件
        return

    if seq[0] == s:  # 找到一种，则
        print(tmp + str(seq[0]))  # 打印

    find2(seq[1:], s, tmp)  # 尾递归 ---不含 seq[0] 的情况
    find2(seq[1:], s - seq[0], str(seq[0]) + '+' + tmp)  # 尾递归 ---含 seq[0] 的情况


# 测试
seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
s = 14  # 和
find2(seq, s)
print()

seq = [11, 23, 6, 31, 8, 9, 15, 20, 24, 14]
s = 40  # 和
find2(seq, s)
