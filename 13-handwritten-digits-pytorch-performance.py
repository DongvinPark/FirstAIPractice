import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    # Check for MPS device
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Transform: Normalize & Convert to Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [0, 255] to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
    ])

    # Load datasets
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=False)

    # Define Model
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = MLP().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Training Loop
    start_time = time.time()
    for epoch in range(5):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print("Total train & test time(sec):", time.time() - start_time)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()