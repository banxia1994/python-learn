#!/usr/bin/python
#coding:utf-8
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
def load(adr):
  ans =pd.read_table(adr,header=None)
  X = ans.iloc[:,:8]
  y = ans[8]
  X.values[:, :]  #转化为dnarray
  y.values[:]
  return X,y

'''
class KNN(object):
  def __init__(self):
    pass

  def train(self,X,y):
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    """
    所谓的预测过程其实就是扫描所有训练集中的图片，计算距离，取最小的距离对应图片的类目
    """
    num_test = X.shape[0]
    # 要保证维度一致哦
    Ypred = np.zeros(len(X))

    # 把训练集扫一遍 -_-||
    for i in xrange(num_test):
      # 计算l1距离，并找到最近的图片
      distances = np.sum(np.abs(self.X_train - X.iloc[i,:]), axis = 1)
      min_index = np.argmin(distances) # 取最近图片的下标
      Ypred[i] = self.y_train[min_index] # 记录下label

    return Ypred
X_train,y_train = load("G:/Pima-training-set.txt")
X_pre,y_pre = load("G:/Pima-prediction-set.txt")
knn = KNN();
knn.train(X_train,y_train)
y = knn.predict(X_pre)
print y==y_pre
accuracy = np.mean(y==y_pre)
print accuracy
'''

import numpy as np
'''
def softmax_loss_naive(W, X, y, reg):

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)    # 得到一个和W同样shape的矩阵
    dW_each = np.zeros_like(W)
    num_train, dim = X.shape
    num_class = W.shape[1]
    f = X.dot(W)    # N by C
    # Considering the Numeric Stability
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # 找到最大值然后减去，这样是为了防止后面的操作会出现数值上的一些偏差
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
    y_trueClass = np.zeros_like(prob)
    y_trueClass[np.arange(num_train), y] = 1.0
    for i in xrange(num_train):
        for j in xrange(num_class):
            loss += -(y_trueClass[i, j] * np.log(prob[i, j]))    # 损失函数的公式L = -(1/N)∑i∑j1(k=yi)log(exp(fk)/∑j exp(fj)) + λR(W)
            dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]#梯度的公式 ∇Wk L = -(1/N)∑i xiT(pi,m-Pm) + 2λWk, where Pk = exp(fk)/∑j exp(fj
        dW += dW_each　　　　　　　　　　　　　　　　　　#这是把每个类的放在了一起
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)  # 加上正则
    dW /= num_traindW += reg * W

    return loss, dW
'''


'''
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.Inputs and outputs
    are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)   # initialize the gradient as zero
    scores = X.dot(W)        # N by C
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores_correct = scores[np.arange(num_train), y]   # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
    margins = scores - scores_correct + 1.0     # N by C
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)
    # compute the gradient
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum
    dW += np.dot(X.T, margins)/num_train + reg * W     # D by C

    return loss, dW

class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
                          batch_size=200, verbose=True):  #注意这里传递的参数设置
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)   # 初始化W

        # Run stochastic gradient descent(Mini-Batch) to optimize W
        loss_history = []
        for it in xrange(num_iters):  #每次随机取batch的数据来进行梯度下降
            X_batch = None
            y_batch = None
            # Sampling with replacement is faster than sampling without replacement.
            sample_index = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[sample_index, :]   # batch_size by D
            y_batch = y[sample_index]      # 1 by batch_size
            # evaluate loss and gradient
            loss, grad = self.loss(X, y, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W += -learning_rate * grad
            if verbose and it % 100 == 0:
                print 'Iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
                  array of length N, and each element is an integer giving the
                  predicted class.
        """
        y_pred = np.zeros(X.shape[1])    # 1 by N
        X=X.T
        y_pred = np.argmax(X.dot(self.W), axis=0) #预测直接找到最后y最大的那个值

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
                   data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass

class LinearSVM(LinearClassifier):
    """
    A subclass that uses the Multiclass SVM loss function
    """
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

class Softmax(LinearClassifier):
    """
    A subclass that uses the Softmax + Cross-entropy loss function
    """
    def loss(self, X_batch, y_batch, reg):
       return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

X_train,y_train = load("G:/Pima-training-set.txt")
X_pre,y_pre = load("G:/Pima-prediction-set.txt")
nn = LinearSVM()
nn.train(X_train,y_train)
y = nn.predict(X_pre)
print y==y_pre
accuracy = np.mean(y==y_pre)
print accuracy
'''
'''
import numpy as np

from matplotlib import pyplot as plt


# train matrix

def get_train_data():
  M1 = np.random.random((100, 2))

  M2 = np.random.random((100, 2)) - 0.7

  plt.plot(M1[:, 0], M1[:, 1], 'ro')

  plt.plot(M2[:, 0], M2[:, 1], 'go')

  return M1, M2


def classify(M1, M2, test_data):
  mean1 = np.mean(M1, axis=0)

  mean2 = np.mean(M2, axis=0)

  mean = (mean1 + mean2) / 2

  # for plot

  km = (mean1[1] - mean2[1]) / (mean1[0] - mean2[0])

  k = km / (-1)

  min_x = np.min(M2)

  max_x = np.max(M1)

  x = np.linspace(min_x, max_x, 100)

  y = k * (x - mean[0]) + mean[1]

  plt.plot(x, y, 'y')

  vector_train = mean1 - mean

  vector_test = test_data - mean

  vector_dot = np.dot(vector_train, vector_test)

  sgn = np.sign(vector_dot)

  return sgn


def get_test_data():
  M = np.random.random((50, 2))

  plt.plot(M[:, 0], M[:, 1], '*y')

  return M


if __name__ == "__main__":

  M1, M2 = get_train_data()

  test_data = get_test_data()

  right_count = 0

  for test_i in test_data:

    classx = classify(M1, M2, test_i)

    if classx == 1:
      right_count += 1

  plt.show()

  print("The accuracy of right classification is %s" % str(right_count / len(test_data)))
'''
  #两层简单的网络
import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)  # 初始化神经网络的参数

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    h1 = np.maximum(0, np.dot(X, W1) + b1)
    #这里是做了一个RELU的activition
    #function
    scores = np.dot(h1, W2) + b2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    scores_max = np.max(scores, axis=1, keepdims=True)  # (N,1)
    # Compute the class probabilities
    exp_scores = np.exp(scores - scores_max)  # (N,C)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N,C)
    # cross-entropy loss and L2-regularization
    correct_logprobs = -np.log(probs[range(N), y])  # (N,1)
    data_loss = np.sum(correct_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss  # 计算出误差
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dscores = probs  # (N,C)
    dscores[range(N), y] -= 1  # 这个是输出的误差敏感项也就是梯度的计算，具体可以看上面softmax 的计算
    dscores /= N
    # Backprop into W2 and b2
    dW2 = np.dot(h1.T, dscores)  # (H,C) BP算法的计算，下面同理
    db2 = np.sum(dscores, axis=0, keepdims=True)  # (1,C
    # Backprop into hidden layer
    dh1 = np.dot(dscores, W2.T)  # (N,H)
    # Backprop into ReLU non-linearity
    dh1[h1 <= 0] = 0
    # Backprop into W1 and b1
    dW1 = np.dot(X.T, dh1)  # (D,H)
    db1 = np.sum(dh1, axis=0, keepdims=True)  # (1,H)
    # Add the regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg * W1
    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      sample_index = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[sample_index, :]  # (batch_size,D)
      y_batch = y[sample_index]  # (1,batch_size)

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      print grads['b2']
      a = grads['b2'].reshape(-1)
      grads['b2'] = a
      a = grads['b1'].reshape(-1)
      grads['b1'] = a
      grads['b1'].reshape(-1)
      v_W2 = - learning_rate * grads['W2']
      self.params['W2'] += v_W2
      self.params['b2'] -= learning_rate * grads['b2']
      v_W1 = - learning_rate * grads['W1']
      self.params['W1'] += v_W1
      v_b1 = - learning_rate * grads['b1']
      self.params['b1'] += v_b1     # 对参数进行更新
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch)
        val_acc = (self.predict(X_val) == y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = None
    h1 = np.maximum(0, (np.dot(X, self.params['W1']) + self.params['b1']))
    scores = np.dot(h1, self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

X_train,y_train = load("G:/Pima-training-set.txt")
X_pre,y_pre = load("G:/Pima-prediction-set.txt")
nn = TwoLayerNet(3,4,2);
nn.train(X_train,y_train,0,0)
y = nn.predict(X_pre)
print y==y_pre
accuracy = np.mean(y==y_pre)
print accuracy