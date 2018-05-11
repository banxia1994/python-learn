#coding:utf-8
import os
from PIL import Image
import glob
import re
import random

dir_left = '/data/Fingervein/MMCBNU-6000/ROI_Left/'
dir_right = '/data/Fingervein/MMCBNU-6000/ROI_Rightresize/'
dir1 = '/data/Fingervein/MMCBNU-6000/'
dir_rec = '/data/Fingervein/MMCBNU-6000/ROI_rectangle/'
dir_sd='/data/Fingervein/Shanda/Nor_database'
f = lambda x: x[x.find('_')+1:].split('.')[0]
d = lambda x: x[x.find('_'):]
s = lambda x: x[0:x.find('.')].split('/')[-1]
a = lambda x: x[0:x.find('.')].split('_')[-1]

def genvalidtxt(dir):

    #dirs = os.listdir(dir)
    dirs = glob.glob(os.path.join(dir,'*.bmp'))
    dirs.sort()

    with open('../valid_utfvp.txt','w') as train:
        for i in dirs:
            #ids = int(i.split('.')[0])
            ids = int(s(i))
            #istr = i.split('/')[-1]
            # train.writelines(istr+' '+ str((ids-1)//10)+'\n') #训练图片的ｔｘｔ
            train.writelines(i+' ' +str((ids-1)//4) + '\n')#测试图片的ｔｘｔ
#genvalidtxt('/data/Fingervein/UTFVP/ROIs_new128x60')

#生成训练的ｔｘｔ
def gentxt(dir):

    #dirs = os.listdir(dir)
    dirs = glob.glob(os.path.join(dir,'*.bmp'))
    dirs.sort()

    with open('../train_sd.txt','w') as train,open('../test_sd.txt','w') as test:
        for i in dirs:
            #ids = int(i.split('.')[0])
            ids = int(s(i))
            if ids%6 == 1:
                test.writelines(i+' '+ str((ids)//6+600)+'\n')#(ids-1)//10
            else:
            #istr = i.split('/')[-1]
            # train.writelines(istr+' '+ str((ids-1)//10)+'\n') #训练图片的ｔｘｔ
                train.writelines(i+' ' +str((ids)//6+600) + '\n')#测试图片的ｔｘｔ
#gentxt(dir_sd)
#gentxt(dir_left)
#gentxt('/data/Fingervein/MMCBNU-6000/ROI_rectangle/')

def genForValidTxt():
    with open('./txt/4_1.txt','w') as v1,open('./txt/4_2.txt','w') as v2:
        with open('../valiaLeft.txt','r') as valid:
            lines = valid.readlines()
            for line in lines:
                ids = int(a(line.split()[0]))
                if ((ids-1)//4)%2==0:
                    v1.writelines(line)
                else:
                    v2.writelines(line)
#genForValidTxt()

#生成测试的对
def genTxtForTestSiamese():
    with open('./txt/4_1.txt','r') as v1,open('./txt/4_2.txt','r') as v2:
        with open('../ValidSiameseLeftNew.txt','w') as valid:
            v1lines = v1.readlines()
            v2lines = v2.readlines()
            for i in v1lines:
                for j in v2lines:
                    if i.split()[-1] == j.split()[-1]:
                        valid.writelines(i.split()[0]+' '+j.split()[0]+'\n')
                    else:
                        if random.random() > 0.99:
                            valid.writelines(i.split()[0]+' '+j.split()[0]+' '+'0'+'\n')
#genTxtForTestSiamese()

#resize 图片
def resizeImg(dir_sd):
    imgs = os.listdir(dir_sd)
    for i in imgs:
        im = Image.open(dir_sd+i)
        path = '/data/Fingervein/UTFVP/ROIs_new128x60/'
        if not os.path.exists(path):
            os.mkdir(path)
        im.resize((128,60),Image.ANTIALIAS).save(path+i,quality = 100)

#resizeImg('../UTFVP/ROIs_new/')

#之前随机生成训练对
import random
def genrandPair(name,num,line_num):
    with open('../{}.txt'.format(name),'w') as pair:
        count = 0
        for  i in range(line_num):
            pair.writelines(str(random.randint(0,num-1))+' '+str(random.randint(0,num-1))+'\n')
            if ((i+1)%3==0):
                pair.writelines(str(count)+' '+str(count)+'\n')
                count += 1
#为训练生成对最准备
def genPiarForSiamese():
    with open('./txt/0_4.txt','w') as a,open('./txt/5_9.txt','w') as b:
        p = re.compile(".*[0-4]$")
        for i in range(6000):
            if p.match(str(i)):
                a.writelines(str(i)+'\n')
            else:
                b.writelines(str(i)+'\n')

#genPiarForSiamese()

#生成训练对
def genSiamesePair():
    with open('./txt/0_4.txt','r') as a,open('./txt/5_9.txt','r') as b,open('../siamesePairNewBalanceNew.txt','w') as c:
        linesa = a.readlines()
        linesb = b.readlines()
        p = re.compile(".*[6]$")
        q = re.compile(".*[3]$")
        for i in linesa:
            for j in linesb:
                if int(i)/10 == int(j)/10:
                    c.writelines(i.strip()+' '+j.strip()+'\n')
                else:
                    ####　随机生成数据对
                    # if random.random() > 0.98:
                    #     c.writelines(i.strip() + ' ' + j.strip() + '\n')
                    if (int(j)/8) % 15 == 0 and q.match(str(i)) and p.match(str(j)):
                        c.writelines(i.strip() + ' ' + j.strip() + '\n')
#genSiamesePair()

#生成测试对
def genTestPair(option): # option =0 mean pairs of left and right , = 1 mean left pair =2 means right pair
    imgs_left = os.listdir(dir1+'ROI_Left')
    imgs_right = os.listdir(dir1+'ROI_Right')
    if option == 0:
        with open('../evapair10.txt','w') as eva:
            for i in imgs_left:
                list1 = [random.randint(0,3487) for _ in range(10)]
                for r in list1:
                    imgR = imgs_right[r]
                    if (int(f(i))-1)/8 == (int(f(imgR))-1)/8:
                        eva.writelines(dir1+'ROI_Left/'+i+' '+dir1+'ROI_Right/'+imgR+'\n')
                    else:
                        eva.writelines(dir1+'ROI_Left/'+i+' '+dir1+'ROI_Right/'+imgR+' '+'0'+'\n')
                eva.writelines(dir1+'ROI_Left/'+i+' '+dir1+'ROI_Right/'+'Right'+d(i)+'\n')
    elif option == 1:
        with open('../evapairLeft.txt','w') as eva:
            for i in imgs_left:
                list1 = [random.randint(0,3487) for _ in range(10)]
                for l in list1:
                    imgL = imgs_left[l]
                    if (int(f(i))-1)/8 == (int(f(imgL))-1)/8:
                        eva.writelines(dir1+'ROI_Leftresize/'+i+' '+dir1+'ROI_Leftresize/'+imgL+'\n')
                    else:
                        eva.writelines(dir1+'ROI_Leftresize/'+i+' '+dir1+'ROI_Leftresize/'+imgL+' '+'0'+'\n')
                eva.writelines(dir1+'ROI_Leftresize/'+i+' '+dir1+'ROI_Leftresize/'+'Left'+d(i)+'\n')
    elif option == 2:
        with open('../evapairRight.txt','w') as eva:
            for i in imgs_right:
                list1 = [random.randint(0,3487) for _ in range(10)]
                for l in list1:
                    imgL = imgs_right[l]
                    if (int(f(i))-1)/8 == (int(f(imgL))-1)/8:
                        eva.writelines(dir1+'ROI_Rightresize/'+i+' '+dir1+'ROI_Rightresize/'+imgL+'\n')
                    else:
                        eva.writelines(dir1+'ROI_Rightresize/'+i+' '+dir1+'ROI_Rightresize/'+imgL+' '+'0'+'\n')
                eva.writelines(dir1+'ROI_Rightresize/'+i+' '+dir1+'ROI_Rightresize/'+'Right'+d(i)+'\n')
    else:
        print 'reset option,option=0 L+R,option=1 L,option=0,R'


#concat ｔｘｔ文件
def concatTxt(txt1,txt2):
    with open('./concatTest.txt','w') as ct:
        with open(txt2,'r') as t2,open(txt1,'r')as t1:
            for line in t1.readlines():
                ct.writelines(line)
            for line in t2.readlines():
                ct.writelines(line)
    ct.close()
#concatTxt('./test_sd.txt','./test.txt')
#打乱txt文件
def shuffleTxt(dir,name):
    with open(dir+name,'r') as ori:
        lines = ori.readlines()
        random.shuffle(lines)
    with open(dir+'pairs_15newsh.txt','w') as tar:
        for i in lines:
            tar.writelines(i)
shuffleTxt('../','pairs_15new.txt')
#shuffleTxt('./','concatTest.txt')
#shuffleTxt('../','siamesePairNewBalanceNew.txt')
#genrandPair('testpair',600,600*3)
#genrandPair('trainpair',5400,5400*3)
#print (random.randint(0,600-1))

#resizeImg()


#print [random.randint(0,3487) for _ in range(20)]
#print f('r_001.bmp')
#genTestPair()

#genTestPair(1)
#shuffleTxt('../','evapairLeft.txt')