#coding:utf-8
import os
import cv2





dir = '/mnt/视频/'
dir_label = '/mnt/视频（图标）/'
dir_frame = '/mnt/图标img/'

lists_mvf = os.listdir(dir)
lists_labf = os.listdir(dir_label)




def AllFrames(dir,FPS=25,gap = 5):
    frames = []
    with open(dir,'r') as f:
        lines = f.readlines()
        for i in lines:
            s = i.split(':')
            if len(s) == 3:
                seconds1 = int(s[0])*3600+int(s[1])*60+int(s[2])
                currentframe = seconds1*FPS
                for f in range(0,FPS,FPS/5):
                    frames.append(currentframe+f)
            elif len(s) == 5:
                els = s[2].split('-')
                seconds1 = int(s[0])*3600+int(s[1])*60+int(els[0])
                seconds2 = int(els[1])*3600+int(s[3])*60+int(s[4])
                if (seconds2 - seconds1) < gap:
                    for st in range(seconds1,seconds2+1):
                        currentframe = st*FPS
                        for f in range(0,FPS,FPS/5):
                            frames.append(currentframe+f)
                else:
                    print 'time between two frames is too long!'
            else:
                print i,'undefined type of label!'
        return frames
def getName(s1,s2):
    o = []
    t = []
    i = 0
    for x in s1:
        i += 1
        if x in s2:
            q = s2.find(x)
            o.append(i-1)
            t.append(q)
    for c in o:
        tar = []
        z = o.index(c)
        z = t[z]
        for s in range(c,len(s1)):
            if s1[s] == s2[z]:
                d = s1[s]
                tar.append(d)
                z += 1
                if z == len(s2):
                    break
            elif len(tar) < 15:
                break
        if len(tar) > 15:
            return ''.join(tar)
    return []

def getFrameJpg():
    for folder_mvf in lists_mvf:
        realName = getName(folder_mvf, lists_labf[0])
        realLabf = lists_labf[0]
        for folder_labf in lists_labf:
            re = getName(folder_labf, folder_mvf)
            if re and len(re) > len(realName):
                realName = re
                realLabf = folder_labf

        if os.path.isdir(dir + folder_mvf):
            files = os.listdir(dir + folder_mvf)
            labels = os.listdir(dir_label + realLabf)
            for f in files:
                if '.' in f:
                    realPicF = getName(f, labels[0])
                    realLabel = labels[0]

                    for l in labels:
                        la = getName(f, l)
                        if la and len(la) > len(realPicF):
                            realPicF = la
                            realLabel = l
                    picpath = dir_frame + realName #+ '/' + realPicF
                    mvpath = dir + folder_mvf + '/' + f
                    labelpath = dir_label + realLabf + '/' + realLabel
                    # if os.path.exists(picpath):
                    #    break
                    if not os.path.exists(picpath):
                        os.mkdir(picpath)
                    try:
                        picpath = dir_frame + realName + '/' + realPicF
                    except Exception,e:
                        print realPicF
                    if os.path.exists(picpath):
                        continue
                    if not os.path.exists(picpath):
                        os.mkdir(picpath)
                    print 'save pics',picpath
                    cap = cv2.VideoCapture(mvpath)
                    fps = int(cap.get(5))
                    fs = AllFrames(labelpath, fps, 5)
                    for i in fs:
                        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)  # frame_id
                        ret, frame = cap.read()
                        cv2.imwrite(os.path.join(picpath + '/', str(i) + '.jpg'), frame)


merge = open(dir_frame+'/Merge.txt','w')
def mergeTxt(dir):
    txtLists = os.listdir(dir)
    for i in txtLists:
        if '.txt' in i:
            writeLine(os.path.join(dir,i),merge)
            continue
        print os.path.join(dir,i)
        if os.path.isdir(os.path.join(dir,i)):
            mergeTxt(os.path.join(dir,i))
    if not hasFolder(dir):
        return



def hasFolder(dir):
    lists = os.listdir(dir)
    count = 0
    for i in lists:
        if os.path.isdir(os.path.join(dir,i)):
            count += 1
    if count == 0:
        return False
    return True

def writeLine(dir_txt,merge):
    lines = open(dir_txt,'r').readlines()
    for line in lines:
        merge.writelines(line)

mergeTxt(dir_frame)








# cap = cv2.VideoCapture('/mnt/视频/10月12日重点新闻合集/lizhiwei法国2：1葡萄牙-全场.MP4')
# fps = int(cap.get(5))
# fs = AllFrames(fps,5)
# for i in fs:
#     cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,i) #frame_id
#     ret,frame = cap.read()
#     cv2.imwrite(os.path.join('/data/testPic/',str(i)+'.jpg'),frame)
