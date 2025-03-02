import numpy as np
import matplotlib.pyplot as plt

# Create a 3D loss surface
W1 = np.linspace(-2, 2, 50)  # Range for weight 1
W2 = np.linspace(-2, 2, 50)  # Range for weight 2
W1, W2 = np.meshgrid(W1, W2)
Loss = (W1**2 + W2**2)  # Example quadratic loss function

# Plot the loss surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Loss, cmap='coolwarm')

ax.set_xlabel('Weight 1')
ax.set_ylabel('Weight 2')
ax.set_zlabel('Loss (MSE)')
ax.set_title('Gradient Descent - Loss Surface')

plt.show()