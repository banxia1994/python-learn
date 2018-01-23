import os, os.path
import shutil, string

dir = "I:\code\data1"
outdir = "data1"
label = " 1"

fileList = os.listdir(dir)


fileinfo = open('list.csv', 'w')


for i in range(200):
    curname = os.path.join(outdir, i)
    print curname
    fileinfo.write(curname + ' 1' + '\n')
    # print i
fileinfo.close()