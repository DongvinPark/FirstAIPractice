import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 2 input, 3 hidden
        self.fc2 = nn.Linear(3, 1)  # 3 hidden, 1 output

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create a network and print output for an example input
net = Net()
sample_input = torch.tensor([[0.0, 1.0]])

# simple network returns tensor
output = net(sample_input)
print(output)