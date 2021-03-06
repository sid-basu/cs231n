import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
    scores = X[i,:].dot(W)
    scores -= np.max(scores)
#    loss += - scores[y[i]] + np.log(np.sum(np.exp(scores)))
    softmaxes = np.exp(scores) / np.sum(np.exp(scores))
    loss -= np.log(softmaxes[y[i]])

    dW += (X[i,:] * softmaxes[:, np.newaxis]).T
    dW[:, y[i]] -= X[i,:]

#  for i in range(num_train):
#    sum_scores_inv = 1.0 / np.sum(W.T.dot(X[i,:]))
#    dW += np.reshape(X[i,:] * sum_scores_inv, (dW.shape[0],1))
#    dW[:,y[i]] -= X[i,:]


  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW +=  2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #calculate scores, loss
  scores = X.dot(W)
  scores -= np.max(scores, axis = 1)[:, np.newaxis]
  softmaxes = np.exp(scores) / np.sum(np.exp(scores), axis = 1)[:, np.newaxis]
  softmaxes_correct = softmaxes[np.arange(softmaxes.shape[0]), y]

  loss = - np.mean(np.log(softmaxes_correct))

  #calculate gradient
  correct_class_weights = np.zeros_like(softmaxes)
  correct_class_weights[np.arange(correct_class_weights.shape[0]), y] = -1

  dW = X.T.dot(softmaxes + correct_class_weights)

  num_train = X.shape[0]
  dW /= num_train

  #incorporate regularization
  loss += reg * np.sum(W * W)
  dW +=  2 * reg * W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

