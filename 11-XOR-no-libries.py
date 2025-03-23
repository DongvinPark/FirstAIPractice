import numpy as np
import random

data = [
    [[0,0],[0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]],
]

iterations = 5000
lr = 0.1  # learning rate
mo = 0.4 # momentum factor

# activation func 1 : sigmoid
def sigmoid(x, derivative=False):
    if(derivative==True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# activation func 1 : tanh
def tanh(x, derivative=False):
    if(derivative==True):
        return 1 - x**2
    return np.tanh(x)

# make weight array matrix
def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill]*j)
    return mat

### Neural Network class
class NeuralNetwork:

    def __init__(self, num_x, num_yh, num_yo, bias=1):
        # input value(num_x),
        # initial value of hidden layer(num_yh),
        # initial value of output layer(num_yo),
        # bias
        self.num_x = num_x + bias
        self.num_yh = num_yh
        self.num_yo = num_yo

        # initial values of activation functions
        self.activation_input = [1.0] * self.num_x
        self.activation_hidden = [1.0]*self.num_yh
        self.activation_out = [1.0] * self.num_yo

        # initial values of weight input
        self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] = random.random()

        # initial values of weight output
        self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                self.weight_out[j][k] = random.random()

        # prev initial values of weight for momentum sigmoid
        self.gradient_in = makeMatrix(self.num_x, self.num_yh)
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)

    def update(self, inputs):

        # activation function of input layer
        for i in range(self.num_x -1):
            self.activation_input[i] = inputs[i]

        # activation function of hidden layer
        for j in range(self.num_yh):
            sum = 0.0
            for i in range(self.num_x):
                sum = sum + self.activation_input[i] * self.weight_in[i][j]
            self.activation_hidden[j] = tanh(sum, False)

        # activation function of output layer
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh):
                sum = sum + self.activation_hidden[j] * self.weight_out[j][k]
            self.activation_out[k] = tanh(sum, False)

        return self.activation_out[:]


    def backPropagate(self, targets):

        # calculate delta output
        output_deltas = [0.0] * self.num_yo
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            output_deltas[k] = tanh(self.activation_out[k], True) * error

        # error function of hidden layer
        hidden_deltas = [0.0]*self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):
                error = error + output_deltas[k] * self.weight_out[j][k]

            hidden_deltas[j] = tanh(self.activation_hidden[j], True) * error

        # update output weight values
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[j]
                v = mo * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient

        # update input weight values
        for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo*self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient

        # calculate error(method of least squares)
        error = 0.0
        for k in range(len(targets)):
            error = (error + 0.5 * (targets[k] - self.activation_out[k]) ** 2)
        return error

    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = (error + self.backPropagate(targets))
            if i%500 == 0:
                print('error: %-.5f' % error)

    def result(self, patterns):
        for p in patterns:
            print('Input: %s, Predict: %s' % (p[0], self.update(p[0]) ))

if __name__ == '__main__':

    # 2 inputs, 2 layers, 1 output.
    n = NeuralNetwork(2,2,1)
    n.train(data)
    n.result(data)