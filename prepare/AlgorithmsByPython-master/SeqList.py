#coding:utf-8
class SeqList:
    def __init__(self,maxcap=10):
        self.max = maxcap
        self.num = 0
        self.data = [None]*self.max

    def is_empty(self):
        return self.num is 0

    def is_full(self):
        return self.num is self.num

    def __getitem__(self,i):
        if 0<=i<self.num:
            return self.data[i]
        else:
            return IndexError

    def __setitem__(self, key, value):
        if 0<=key<self.num:
            self.data[key] = value
        else:
            raise IndexError

    def getLoc(self,value):
        n=0
        for i in range(self.num):
            if self.data[i] == value:
                return i
        if i == self.num:
            return -1

    def count(self):
        return self.num

    def appendlast(self,value):
        if self.num >=self.max:
            print 'list is full'
            return
        else:
            self.data[self.num] = value
            self.num+=1

    def insert(self,i,value):
        if i<0 or i>=self.num:
            raise IndexError
        else:
            for j in range(self.num,i,-1):
                self.data[j] = self.data[j-1]
            self.data[i] = value
            self.num +=1

    def remove(self,i):
        if i<0 or i>=self.num:
            raise IndexError
        else:
            for j in range(i,self.num):
                self.data[j] = self.data[j+1]
            self.num-=1

    def prin(self):
        for i in range(0,self.num):
            print self.data[i]

    def des(self):
        self.__init__()

a = SeqList()
a.appendlast(1)
a.appendlast(2)
a.insert(1,5)
a.__setitem__(1,2)
#print a.getLoc(1)

#print a.count()
a.prin()


