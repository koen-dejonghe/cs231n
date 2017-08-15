import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero


  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class = y[i]
    correct_class_score = scores[correct_class]
    num_classes_contributing_to_loss = 0

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        num_classes_contributing_to_loss += 1

    dW[:, correct_class] -= X[i] * num_classes_contributing_to_loss

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def _svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W) # NxC

  one_hot_y = np.zeros((num_train, num_classes), "int32")
  one_hot_y[np.arange(num_train), y] = 1
  incorrect_class_scores = scores[one_hot_y == 0].reshape((num_train, -1))

  correct_class_scores = scores[np.arange(num_train), y] # Nx1

  margins = incorrect_class_scores.T - correct_class_scores.T + 1 # 9xN
  loss = np.sum(np.maximum(0, margins))

  loss /= num_train
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

# https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  # return svm_loss_naive(W, X, y, reg)

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #############################################################################
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  yi_scores = scores[np.arange(num_train), y] # Nx1
  # yi_scores = scores[np.arange(scores.shape[0]),y] # http://stackoverflow.com/a/23435843/459241 
  margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
  margins[np.arange(num_train),y] = 0
  loss = np.mean(np.sum(margins, axis=1))
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  binary = margins
  binary[margins > 0] = 1
  row_sum = np.sum(binary, axis=1)
  binary[np.arange(num_train), y] = -row_sum.T
  dW = np.dot(X.T, binary)

  # Average
  dW /= num_train

  # Regularize
  dW += reg*W

  return loss, dW

