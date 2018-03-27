#coding:utf-8
from pypinyin import lazy_pinyin
import sys

Shname = sys.argv[1]
wrFile = sys.argv[2]


f = open(Shname+'.sh','w')
f.writelines('#! /bin/sh'+'\n')

for line in (open(wrFile,'r')):
	line  = line.split()[0]
	lineP = ''.join(lazy_pinyin(unicode(line,'utf-8')))
	#print 'pthon3'+' '+'spiderbaidu.py'
	f.writelines('python3'+' '+'multiSpider.py'+' '+line+' '+lineP.encode('utf-8') +'\n')

