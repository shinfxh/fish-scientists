#!/usr/bin/env python3
"""
Convolutional Neural Network for MNIST Classification
This script trains a simple CNN on the MNIST dataset and evaluates its performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from pathlib import Path

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for MNIST classification"""
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        # First convolutional layer: 1 input channel, 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer: 32 input channels, 64 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout layer with specified dropout rate
        self.dropout = nn.Dropout(dropout_rate)
        # First fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer - 10 classes for digits 0-9
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv1(x)))
        # Second convolutional block: conv -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))
        # Reshape the tensor for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)
        # First fully connected layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # Output layer
        x = self.fc2(x)
        return x

def load_data(batch_size=64):
    """Load MNIST dataset for training and testing"""
    # Define transformations for the training data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
    ])
    
    # Define transformations for the test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load the training data
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Download and load the test data
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, train_loader, optimizer, criterion, device, epoch):
    """Train the model for one epoch"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {train_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%')
    
    return train_loss / len(train_loader), correct / total

def test(model, test_loader, criterion, device):
    """Evaluate the model on the test set"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Update metrics
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / total
    
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {100.*accuracy:.2f}%')
    
    return test_loss, accuracy

def run_baseline_experiment(config):
    """Run baseline experiment with default hyperparameters"""
    print("\n=== Running Baseline Experiment ===")
    train_loader, test_loader = load_data(config["batch_size"])
    
    model = SimpleCNN(dropout_rate=config["dropout_rate"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    elapsed_time = time.time() - start_time
    
    results = {
        "config": config,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accuracies": train_accuracies,
        "test_accuracies": test_accuracies,
        "final_test_accuracy": test_accuracies[-1],
        "training_time": elapsed_time
    }
    
    print(f"\nBaseline experiment completed in {elapsed_time:.2f} seconds")
    print(f"Final test accuracy: {100*results['final_test_accuracy']:.2f}%")
    
    return results

def main():
    """Main function to run the baseline experiment"""
    # Default configuration
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 5,
        "dropout_rate": 0.5
    }
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Run baseline experiment
    results = run_baseline_experiment(config)
    
    print("\nExperiment completed successfully!")
    print("This script can be extended to perform hyperparameter tuning and ablation studies.")

if __name__ == "__main__":
    main() 