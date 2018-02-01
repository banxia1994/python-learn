#coding:utf-8
# 归并排序具体工作原理如下（假设序列共有n个元素）：
# 将序列每相邻两个数字进行归并操作（merge)，形成floor(n/2)个序列，排序后每个序列包含两个元素
# 将上述序列再次归并，形成floor(n/4)个序列，每个序列包含四个元素
# 重复步骤2，直到所有元素排序完毕

def mergeSort(alist):
    if len(alist) > 1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = 0; j = 0; k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i += 1
            else:
                alist[k] = righthalf[j]
                j += 1
            k += 1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i += 1
            k += 1
        while j < len(righthalf):
            alist[k] = righthalf[j]
            j += 1
            k += 1

alist = [54,26,93,17,77,31,44,55,20]
#mergeSort(alist)
print(alist)



def mergeSort2(alist):
    if len(alist)<2:
        return alist
    mid = len(alist)//2
    left = alist[:mid]
    right = alist[mid:]

    left = mergeSort2(left)
    right = mergeSort2(right)

    l,r=0,0
    result = []
    while l<len(left) and r <len(right):
        if left[l] < right[r]:
            result.append(left[l])
            l +=1
        else:
            result.append(right[r])
            r +=1
    result += left[l:]
    result += right[r:]

    return result

print(mergeSort2(alist))
