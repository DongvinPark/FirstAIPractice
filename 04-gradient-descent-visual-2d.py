import numpy as np
import matplotlib.pyplot as plt

# simulate gradient desent
learning_rate = 0.2 # learning_rate matters.
# too small > too long to converge. too big >> Jump around and never converges.

epochs = 10
w = 2 # strat at a high weight
b = 2 # start at a high bias
history = []

for i in range(epochs):
    grad = 2 * w # derivative of loss function (L = w^2)
    w -= learning_rate * grad # update weight
    history.append(w) # store history for plotting

# plot gradient descent steps
plt.figure(figsize=(8,6))
plt.plot(range(epochs), history, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Weight Value")
plt.title("Gradient Descent - Step by Step")
plt.grid()
plt.show()