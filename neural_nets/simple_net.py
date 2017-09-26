import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_dataset():
    from mnist import MNIST
    mndata = MNIST('./python-mnist/data/')
    xtr, ytr = mndata.load_training()
    xte, yte = mndata.load_testing()
    return np.asarray(xtr), np.asarray(ytr), np.asarray(xte), np.asarray(yte)


def generate_data(N, D, K, display):
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    if display == True:
        visualize_data(X, y)
    return N, D, K, X, y


def visualize_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def learn_net(N, D, K, xtr, ytr, xte, yte):
    # initialize parameters randomly
    h = 100  # size of hidden layer
    W = 0.01 * np.random.randn(D, h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.randn(h, K)
    b2 = np.zeros((1, K))

    # some hyperparameters
    step_size = 1e-3
    reg = 1e-3  # regularization strength

    # gradient descent loop
    num_examples = xtr.shape[0]
    for i in xrange(10000):

        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(xtr, W) + b)  # note, ReLU activation
        scores = np.dot(hidden_layer, W2) + b2
        #print("minmax in scores: ", np.min(scores), np.max(scores))

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
        #print(probs)
        #print("probs sum: ", np.sum(probs, axis=1))

        # compute the loss: average cross-entropy loss and regularization
        corect_logprobs = -np.log(probs[range(num_examples), ytr])
        data_loss = np.sum(corect_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
        loss = data_loss + reg_loss
        if i % 100 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), ytr] -= 1
        dscores /= num_examples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(xtr.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

        if i % 500 == 0:
            hidden_layer = np.maximum(0, np.dot(xte, W) + b)
            scores = np.dot(hidden_layer, W2) + b2
            predicted_class = np.argmax(scores, axis=1)
            print('training accuracy: %.5f' % (np.mean(predicted_class == yte)))


#N, D, K, xtr, ytr = generate_data(100, 2, 3, display=False)
#N, D, K, xte, yte = generate_data(100, 2, 3, display=False)
xtr, ytr, xte, yte = load_dataset()
#xtr = xtr.reshape(60000, 28, 28)
print(xtr.shape)
print(ytr.shape)
print(np.min(ytr), np.max(ytr))
N = xtr.shape[0]
D = xtr.shape[1]
K = 10
#plt.imshow(xtr[0], cmap=cm.binary)
#plt.show()
learn_net(N, D, K, xtr, ytr, xte, yte)