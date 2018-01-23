def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
   labelMat = mat(classLabels).transpose() #convert to NumPy matrix
   m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
   for k in range(maxCycles):              #heavy on matrix operations
      h = sigmoid(dataMatrix*weights)     #matrix mult
       error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
   return weights



def stocGradAscent0(dataMatrix, classLabels):
   m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
       error = classLabels[i] - h
       weights = weights + alpha * error * dataMatrix[i]
   return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
   weights = ones(n)   #initialize to all ones
  for j in range(numIter):
     dataIndex = range(m)
      for i in range(m):
         alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
           h = sigmoid(sum(dataMatrix[randIndex]*weights))
           error = classLabels[randIndex] - h
           weights = weights + alpha * error * dataMatrix[randIndex]
          del(dataIndex[randIndex])#删除所选的样本
   return weights

