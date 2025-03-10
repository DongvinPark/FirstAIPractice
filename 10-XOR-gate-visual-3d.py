import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility. This will ensure getting same graph on every execution.
torch.manual_seed(42)

# XOR dataset
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define a simple MLP with one hidden layer
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden = nn.Linear(2, 3)  # Hidden layer projects to 3D
        self.output = nn.Linear(3, 1)  # Output layer

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))  # Activation function in hidden layer
        y = torch.sigmoid(self.output(h))  # Activation function in output layer
        return y, h  # Also return hidden layer output for visualization

# Model, loss, and optimizer
model = XOR_MLP()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred, _ = model(X)
    loss = criterion(y_pred, Y)
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Get hidden layer outputs for visualization
_, hidden_outputs = model(X)
hidden_outputs = hidden_outputs.detach().numpy()

# 3D Visualization of the hidden layer's transformation
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# Plot transformed points
for i in range(4):
    ax.scatter(hidden_outputs[i, 0], hidden_outputs[i, 1], hidden_outputs[i, 2],
               c='r' if Y[i] == 1 else 'b', s=100)

ax.set_xlabel('Hidden Dimension 1')
ax.set_ylabel('Hidden Dimension 2')
ax.set_zlabel('Hidden Dimension 3')
ax.set_title('XOR Transformed to 3D Space')
plt.show()

# Graph Analysis
# Red dot means 'True' result on XOR truth table.
# Blue dot means 'False' result on XOR truth table.
# The hidden layer transforms the original 2D XOR inputs into 3D space.
# In the higher-dimensional space, the red and blue dots are linearly separable.