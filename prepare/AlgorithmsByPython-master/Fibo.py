def printFibo(num):

    a = 0
    b = 1
    i = 0

    while i < num:
        print a
        a,b = b, a+b
        i += 1
printFibo(15)

def fibo(num):
    if num == 0:
        res = 0
    elif num ==1:
        res = 1
    else:
        res = fibo(num-1)+fibo(num-2)
    return res

def print_fibo(num):
    for i in range(num):
        print fibo(i)

print print_fibo(15)

def fibo1(num):
    x,y,n = 0,1,0

    while n <num:
        print x
        x,y,n = y,x+y,n+1
fibo1(15)