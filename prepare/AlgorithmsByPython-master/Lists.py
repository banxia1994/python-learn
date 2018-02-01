#coding:utf-8
# class Node:
#     def __init__(self, initdata):
#         self.data = initdata
#         self.next = None
#
#     def getData(self):
#         return self.data
#
#     def getNext(self):
#         return self.next
#
#     def setData(self, newdata):
#         self.next = newdata
#
#     def setNext(self, nextNode):
#         self.next = nextNode
#
#
# temp = Node(93)
# temp.setData(10)
# print(temp.getNext())
#
# # 定义一个无序链表
# class UnorderedList:
#     def __init__(self):
#         self.head = None
#
#     def isEmpty(self):
#         return self.head == None
#
#     def add(self, item):
#         temp = Node(item)
#         temp.setNext(self.head)
#         self.head = temp
#
#     def size(self):
#         current = self.head
#         count = 0
#         while current != None:
#             count += 1
#             current = current.getNext()
#         return count
#
#     def search(self, item):
#         current = self.head
#         found = False
#         while current != None and not found:
#             if current.getData() == item:
#                 found = True
#             else:
#                 current = current.getNext()
#         return found
#
#     def remove(self, item):
#         current = self.head
#         previous = None
#         found = False
#         while not found:
#             if current.getData() == item:
#                 found = True
#             else:
#                 previous = current
#                 current = current.getNext()
#
#         if previous == None:
#             self.head = current.getNext()
#         else:
#             previous.setNext(current.getNext())
#
# myList = UnorderedList()
# myList.add(31)
# myList.add(77)
# myList.add(17)
# myList.add(93)
# myList.add(26)
# myList.add(54)
# print(myList.search(17))
# myList.remove(54)
# print(myList.search(54))
class Node:
    def __init__(self,initdata):
        self.data = initdata
        self.next = None

    def getdata(self):
        return self.data

    def setdata(self,data):
        self.data = data

    def getnext(self):
        return self.next

    def setnext(self,node):
        self.next = node
# newNode = Node(5)
#
# print newNode.getdata()
class List:
    def __init__(self):
        self.head = None

    def isEmpty(self):
        return self.head is None

    def size(self):
        if self.isEmpty():
            return 0
        else:
            current = self.head
            count = 0
            while current!=None:
                count+=1
                current = current.getnext()
            return count

    def add(self,data): #在链表前面添加数据
        temp = Node(data)
        temp.setnext(self.head)
        self.head = temp

    def append(self,data): #在链表后添加单元
        temp = Node(data)
        if self.isEmpty():
            self.head = temp
        else:
            current = self.head
            while current.getnext() != None:
                current = current.getnext()
            current.setnext(temp)

    def search(self,data):
        if self.isEmpty():
            return False
        else:
            current = self.head
            while current != None:
                if current.getdata() == data:
                    return True
                current = current.getnext()
            return False

    def remove(self,data):
        if self.isEmpty():
            return ValueError,'Lisr is empty'
        current = self.head
        pre = None
        while current != None:
            if current.getdata() == data:
                if not pre:
                    self.head=current.getnext()
                else:
                    pre.setnext(current.getnext())
                break
            else:
                pre = current
                current = current.getnext()

    def insert(self,pos,data):
        if pos > self.size():
            self.append(data)
        else:
            temp = Node(data)
            count = 1
            pre = None
            current = self.head
            while count != pos:
                count+=1
                pre = current
                current = current.getnext()
            if not pre:
                self.add(data)
            else:
                pre.setnext(temp)
                pre.getnext().setnext(current)
myList = List()
myList.append(1)
myList.add(31)
myList.add(77)
myList.add(17)
myList.add(93)
myList.add(26)
myList.add(54)
print(myList.search(17))
myList.remove(54)
print myList.size()
myList.insert(5,11)
myList.add(0)
print(myList.search(54))

