---
layout: post
title: "Convolutional Neural Networks with TensorFlow"
date: 2017-05-04
excerpt: "A cursory overview of CNNs with a practical example using MNIST."
tags: [python, neural networks, CNNs, deep learning, TensorFlow]
comments: true
---
This module describes how a convolutional neural network works, and we will demonstrate its application on the MNIST dataset using TensorFlow. 

Convolutional neural networks (CNNs) are a type of feed-forward artificial neural network whose neuron interconnectivity emulates that of the animal visual cortex. CNNs are particularly useful with computer vision tasks such as image classification; however, they can be applied in other machine learning tasks as long as the ordering of the attributes along at least one of the dimensions is essential for classification. For example, CNNs can also be used in natural language processing and sound analytics.  

The primary components of CNNs are the convolutional layer, the pooling layer, the ReLU layer, and the fully connected layer.  

### Convolutional layer  

The convolutional layer begins with a 3-dimensional version of the original input, generally an image with dimensions of color, width, and height. Then, the image is broken into a subset of filters, also known as kernels, each with a smaller receptive fields than the overall image. These filters are then convolved across the width and height of the input volume, computing the dot product between the entries of the filter and the input and producing a 2-dimensional activation map of that filter. As a result, the network learns filters that activate when it detects some specific type of feature at some spatial position in the input. The fact that a filter is dragged across the entire image allows CNNs to have translational invariance, meaning they deal well with objects located in different parts of the image.  

The activation maps are then stacked to form the depth of the output of the convolution layer. Every entry in the output volume can thus also be interpreted as an output of a neuron that looks at a small region in the input and shares parameters with neurons in the same activation map.

A key concept here is local connectivity, which indicates that each neuron is connected to only a small region of the input volume.  The size of the filter, also known as the receptive field, is a key factor in determining the extent of the connectivity.  

Other key parameters are depth, stride, and padding. Depth indicates how many feature maps we are building. Stride controls the step of each filter as it moves across and down the image. Generally, a step of one is used, leading to heavily overlapped receptive fields and large output volumes. Padding allows us to control the spatial size of the output volume. If we add zero-padding, it will give us an output width and height that is the same size as the input volume.

### Pooling layer  

Pooling is a form of non-linear down-sampling that allows us to shrink the output of convolution while preserving the most prominent features. The most common method of pooling is max pooling, where the input image, in this case the activation map from the convolution layer, is partitioned into non-overlapping rectanges, from which the maximim value is taken.  

One of the key advantages of pooling is that it reduces the number of parameters and the amount of computation in the network, thereby reducing overfitting. Additionally, because pooling takes away information about the exact location of a particular feature in favor of its position relative to other features, it in turn offers an additional form of translational invariance.  

The most common pooling size is 2-by-2 with a stride of 2, thereby eliminating 75% of the activations from the input map. 

### ReLU layer  

The Rectifier Linear Unit layer applies the activation function $f(x)= max(0,x)$ to the output of the pooling layer. It increases the nonlinear properties of the decision function and of the overall network without affecting the receptive fields of the convolution layer. Of course, we can also apply other standard non linear activation functions, such as _tanh_ or the _sigmoid_ function.  

### Fully connected layer  

Following the ReLU layer, we take its output and flatten it to a vector that will serve to tune our weights in a standard neural network. 

### CNN with TensorFlow on the MNIST data set  

Here, we will show how we can achieve almost 99% accuracy on the MNIST data set using CNN with TensorFlow. 

First, we import all the necessary libraries.


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os

from datetime import datetime
from sklearn.utils import shuffle
```

Our basic helper functions will give us the error rate and the indicator matrix for our predictions.


```python
def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)
```

Next, we load the data, normalize it, reshape it, and generate our train and test datasets.


```python
data = pd.read_csv(os.path.join('Data', 'train.csv'))
```


```python
def get_normalized_data(data):
    data = data.as_matrix().astype(np.float32)
    np.random.shuffle(data)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std
    Y = data[:, 0]
    return X, Y
```


```python
X, Y = get_normalized_data(data)
```


```python
X = X.reshape(len(X), 28, 28, 1)
```


```python
X = X.astype(np.float32)
```


```python
Xtrain = X[:-1000,]
Ytrain = Y[:-1000]
Xtest  = X[-1000:,]
Ytest  = Y[-1000:]
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)
```

In our convpool function, we will take a stride of one, and we will ensure that the dimensions of output of the convolution are the same as the input by setting _padding_ to 'SAME.' Our downnsampling will be of size two, and we will apply the ReLu activation function on the output.


```python
def convpool(X, W, b):
    # just assume pool size is (2,2) because we need to augment it with 1s
    # - stride is the interval at which to apply the convolution
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(pool_out)
```

The way we always initialize weights is random normal / sqrt(fan in + fan out). The key point is it's random with a variance restricted by the size.


```python
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))
    return w.astype(np.float32)
```

We define our gradient descent parameters, which include the number of iterations, batch size, number of hidden layers, number of classes, and the pool size.


```python
# gradient descent params
max_iter = 6
print_period = 10
N = Xtrain.shape[0]
batch_sz = 500
n_batches = N / batch_sz

# limit samples since input will always have to be same size
# you could also just do N = N / batch_sz * batch_sz

M = 500
K = 10
poolsz = (2, 2)
```

When initializing our filters, we have to remember that TensorFlow has its own ordering of dimensions. The output after convpooling is going to be 7x7, which is different from Theano.


```python
W1_shape = (5, 5, 1, 20) # (filter_width, filter_height, num_color_channels, num_feature_maps)
W1_init = init_filter(W1_shape, poolsz)
b1_init = np.zeros(W1_shape[-1], dtype=np.float32) # one bias per output feature map

W2_shape = (5, 5, 20, 50) # (filter_width, filter_height, old_num_feature_maps, num_feature_maps)
W2_init = init_filter(W2_shape, poolsz)
b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

W3_init = np.random.randn(W2_shape[-1]*7*7, M) / np.sqrt(W2_shape[-1]*7*7 + M)
b3_init = np.zeros(M, dtype=np.float32)
W4_init = np.random.randn(M, K) / np.sqrt(M + K)
b4_init = np.zeros(K, dtype=np.float32)
```

Next, we define our input and target placeholders, as well as the variables which will be updated during the training process. 


```python
# using None as the first shape element takes up too much RAM unfortunately
X = tf.placeholder(tf.float32, shape=(batch_sz, 28, 28, 1), name='X')
T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')
W1 = tf.Variable(W1_init.astype(np.float32))
b1 = tf.Variable(b1_init.astype(np.float32))
W2 = tf.Variable(W2_init.astype(np.float32))
b2 = tf.Variable(b2_init.astype(np.float32))
W3 = tf.Variable(W3_init.astype(np.float32))
b3 = tf.Variable(b3_init.astype(np.float32))
W4 = tf.Variable(W4_init.astype(np.float32))
b4 = tf.Variable(b4_init.astype(np.float32))
```

This is our feedforward mechanism. Note that flattening the output of our second convpool layer requires an extra step when using TensorFlow. We will also apply RMSProp during training in order to accelerate our process of gradient descent.


```python
Z1 = convpool(X, W1, b1)
Z2 = convpool(Z1, W2, b2)
Z2_shape = Z2.get_shape().as_list()
Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])
Z3 = tf.nn.relu( tf.matmul(Z2r, W3) + b3 )
Yish = tf.matmul(Z3, W4) + b4

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = Yish, labels = T))

train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

# we'll use this to calculate the error rate
predict_op = tf.argmax(Yish, 1)
```

Our training process is standard, except that when making predictions against the test set, due to RAM limitations we need to have a fixed size input; so as a result, we have have to add a slightly complex total cost and prediction computation.


```python
t0 = datetime.now()
LL = []
init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(init)

    for i in range(int(max_iter)):
        for j in range(int(n_batches)):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

            if len(Xbatch) == batch_sz:
                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = 0
                    prediction = np.zeros(len(Xtest))
                    for k in range(int(len(Xtest) / batch_sz)):
                        Xtestbatch = Xtest[k*batch_sz:(k*batch_sz + batch_sz),]
                        Ytestbatch = Ytest_ind[k*batch_sz:(k*batch_sz + batch_sz),]
                        test_cost += session.run(cost, feed_dict={X: Xtestbatch, T: Ytestbatch})
                        prediction[k*batch_sz:(k*batch_sz + batch_sz)] = session.run(
                            predict_op, feed_dict={X: Xtestbatch})
                    err = error_rate(prediction, Ytest)
                    if j == 0:
                        print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
print("Elapsed time:", (datetime.now() - t0))
plt.plot(LL)
plt.show()
```

    Cost / err at iteration i=0, j=0: 2243.417 / 0.805
    Cost / err at iteration i=1, j=0: 116.821 / 0.035
    Cost / err at iteration i=2, j=0: 78.144 / 0.029
    Cost / err at iteration i=3, j=0: 57.462 / 0.018
    Cost / err at iteration i=4, j=0: 52.477 / 0.015
    Cost / err at iteration i=5, j=0: 48.527 / 0.018
    Elapsed time: 0:09:16.157494
    


<img src="/assets/img/Deep%20Learning%20Convolution%20in%20TensorFlow_31_1.png" />


### Conclusion  

As we can see from the results, the model performs with an accuracy between 98% and 99% on the test set. In this exercise, we did not do any hyperparameter tuning, but that would be a natural next step in our process. We can also add regularization, or momentum, or droput. 

### References  
 - <https://en.wikipedia.org/wiki/Convolutional_neural_network>
 - <http://deeplearning.net/tutorial/lenet.html>
 - <https://www.udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow/>
