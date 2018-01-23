# coding:utf-8
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series,DataFrame
from PIL import Image
'''
def getXy(address):
    data = pd.read_table(address, header=None, dtype=str, na_filter=False)
    X = data.iloc[0:, :8]
    y = data[8]
    #W = data_predict.iloc[0:, :8]
    #z = data_predict[8]
    return X,y
feature,y = getXy("G:/Pima-training-set.txt")
X=feature
pca = PCA(n_components=2)
pca.fit(X)
X_new = pca.transform(X)
print(pca.explained_variance_ratio_)
print X_new
XX = pca.inverse_transform(X_new)
p2 = plt.subplot(122)
p2.plot(XX[:,0],XX[:,1],'*')
plt.show()
'''
'''
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片

    width,height = im.size
    im = im.convert("L")
  #  data = im.getdata()
    data = np.array(im,dtype='float')/255.0
    a =  data.shape
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
    new_im = Image.fromarray(new_data)
    # 显示图片
    new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



filename = 'I:/4.jpg'
data = ImageToMatrix(filename)
print data
new_im = MatrixToImage(data)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')

new_im.save('lena_1.jpg')
'''

import cv2

im = Image.open('I:/4.jpg')
im.resize((224,224),Image.ANTIALIAS).save('I:/data/5.jpg',quality = 100)


