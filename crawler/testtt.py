#coding:utf-8
s2 = 'zjl 图标 第19轮录播：皇家马德里VS比利亚雷亚尔'
s1= u'/mnt/图标img/10月12日重点新闻合集/法国2：1葡萄牙-全场'

s3 = ''
for i in range(0,len(s1.split('/'))-1):
    s3 = s3+''.join(s1.split('/')[i])+'/'
s3 = s3 + s1.split('/')[-1] +'.txt'
print s3
print type(s3)
print type(s1.split('/')[-1])

# print type(s1)
#
# s2 = s1.encode('utf-8')
#
# print type(s2)
#
# print s2

#


# def getName():
#     o = []
#     t = []
#     i = 0
#     for x in s1:
#         i += 1
#         if x in s2:
#             q = s2.find(x)
#             o.append(i-1)
#             t.append(q)
#     for c in o:
#         tar = []
#         z = o.index(c)
#         z = t[z]
#         for s in range(c,len(s1)):
#             if s1[s] == s2[z]:
#                 d = s1[s]
#                 tar.append(d)
#                 z += 1
#                 if z == len(s2):
#                     break
#             elif len(tar) < 15:
#                 break
#         if len(tar) > 15:
#             return ''.join(tar)
#     return []
#
# print getName()