import numpy as np

# basic artificial neural network without training.
# simulates AND gate using random weight and bias

# input
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# expected output of AND gate
Y = np.array([[0],[0],[0],[1]])

W = np.random.rand(2,1)
b = np.random.rand(1)
print("Weight Matrix : ")
print(W)
print("\nBias : ")
print(b)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

                  # compute the weight sum     # and add bias
output = sigmoid(      np.dot(X, W)         +     b )
print("\nResult(changes randomly)...")
print(output)