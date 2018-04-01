import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy
import seaborn as sns

def create_placeholders(n_x, n_y):

    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))
    
    return X, Y


def initialize_parameters():

        
    W1 = tf.get_variable("W1", [200,200], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [200,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [100,200], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [100,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [9,100], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [9,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def initialize_parameters(layer_dims):

    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        
#        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*  np.sqrt(2 / layer_dims[l-1])
        
        parameters['W' + str(l)] = tf.get_variable('W' + str(l), [layer_dims[l],layer_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer())
        
        parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l], 1], initializer = tf.zeros_initializer())
        
#        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)
        
    return parameters



def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def compute_cost(Z3, Y):

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):

    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 1                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z3 = forward_propagation(X, parameters)

    cost = compute_cost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        # plot the cost
        plt.plot(np.squeeze(costs[50:]))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters


def read_data(addr, xname, yname):
    data = scipy.io.loadmat(addr)
    x = data[xname]
    y = data[yname]-1
    return x, y

def onehot(y):
    enc = OneHotEncoder()
    enc.fit(y.T)
    y = enc.transform(y.T).toarray().T
    return y

def splitdata(x, y):
    cnt = np.zeros(16, dtype=int)
    for i in range(y.size):
        cnt[y[:, i]] += 1
    srt = -np.sort(-cnt)[:9]
    srt = srt[:9]
    cum = 0
    x_train = np.zeros((x.shape[0], 0))
    x_test = np.zeros((x.shape[0], 0))
    y_train = np.zeros((y.shape[0], 0), dtype=int)
    y_test = np.zeros((y.shape[0], 0), dtype=int)
    for i in range(cnt.size):
        cum += cnt[i]
        if cnt[i] not in srt:
            continue
        xtrain, xtest, ytrain, ytest = train_test_split(x[:, cum-cnt[i]:cum].T, y[:, cum-cnt[i]:cum].T, test_size=0.5)
        x_train = np.append(x_train, xtrain.T, axis=1)
        x_test = np.append(x_test, xtest.T, axis=1)
        y_train = np.append(y_train, ytrain.T, axis=1)
        y_test = np.append(y_test, ytest.T, axis=1)
    
    y_train = onehot(y_train)
    y_test = onehot(y_test)
    
    return x_train, x_test, y_train, y_test




x, y = read_data("datalab.mat", "clsAll", "label")
x = x-np.mean(x) / np.std(x)**2
x_train, x_test, y_train, y_test = splitdata(x, y)
parameters = model(x_train, y_train, x_test, y_test, num_epochs = 3000, minibatch_size=512)