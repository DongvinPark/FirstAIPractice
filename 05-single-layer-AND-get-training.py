import numpy as np

# initialize input X and expected output Y for the AND gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[0],[0],[1]]) # AND gate expected outputs(== right answer)

# randomly initialize weights and bias
np.random.seed(42) # for reproducibility
W = np.random.rand(2,1)
b = np.random.rand(1)

# define sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x) # Sigmoid' = sigmoid(x) X (1 - sigmoid(x))

# define training parameter
learning_rate = 0.1 # how fase the weights update
epochs = 10000 # how many times the model learns

# training loop
for epoch in range(epochs):
    # compute the output
    z = np.dot(X, W) + b # weight sum
    output = sigmoid(z) # apply activation function. output is 4 X 1 matrix

    # compute error
    error = Y - output

    # compute gradients using back-propagation
    d_output = error * sigmoid_derivative(output)

    # update weights. matrix transformation is done by 'X.T'.
    # original X is 4 X 2 matrix. it must be transformed into 2 X 4
    # to apply dot product with output(4 X 1) matrix.
    W += np.dot(X.T, d_output) * learning_rate
    b += ( np.sum(d_output) * learning_rate ) # update bias

    # print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(error**2) # Mean Squared Error
        print(f"Epoch {epoch}, Loss : {loss:.6f}")


# final trained weights and bias
print("\nFinal Weights (W):\n", W)
print("Final Bias (b): ", b)

# test the trained model
print("\nFinal Output:")
final_output = sigmoid(np.dot(X, W) + b) # computer the final output after training
for row in final_output:
    print(f"[{row[0]:.6f}]")
















