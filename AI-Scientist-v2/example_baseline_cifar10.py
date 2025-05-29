#!/usr/bin/env python3
"""
Example baseline CIFAR-10 classifier for ablation studies.

This is a simple CNN baseline that achieves ~72% accuracy on CIFAR-10.
It serves as a starting point for hyperparameter tuning and ablation studies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os
import json
from typing import Dict, Any


class SimpleCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification"""
    
    def __init__(self, num_classes=10, dropout_rate=0.5, hidden_size=512):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Conv blocks with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_data_loaders(batch_size=128, data_augmentation=True):
    """Create CIFAR-10 data loaders"""
    
    # Base transforms
    if data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
    """Train the model"""
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'Acc: {100 * correct / total:.2f}%')
                running_loss = 0.0
        
        epoch_time = time.time() - start_time
        epoch_acc = 100 * correct / total
        
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s - '
              f'Training Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies


def evaluate_model(model, testloader, criterion, device):
    """Evaluate the model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    test_loss = test_loss / len(testloader)
    
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')
    
    return test_accuracy, test_loss


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Main training and evaluation function"""
    
    # Hyperparameters (these can be modified for ablation studies)
    config = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 20,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
        'hidden_size': 512,
        'data_augmentation': True,
        'optimizer': 'Adam',  # Options: 'Adam', 'SGD', 'AdamW'
        'scheduler': 'StepLR',  # Options: 'StepLR', 'CosineAnnealingLR', None
        'step_size': 10,
        'gamma': 0.1,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loaders
    trainloader, testloader = get_data_loaders(
        batch_size=config['batch_size'],
        data_augmentation=config['data_augmentation']
    )
    
    # Model
    model = SimpleCNN(
        dropout_rate=config['dropout_rate'],
        hidden_size=config['hidden_size']
    ).to(device)
    
    print(f'Model has {count_parameters(model):,} trainable parameters')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    # Learning rate scheduler
    if config['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    elif config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
    else:
        scheduler = None
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    train_losses, train_accuracies = train_model(
        model, trainloader, criterion, optimizer, device, config['epochs']
    )
    
    # Update learning rate
    if scheduler:
        for epoch in range(config['epochs']):
            scheduler.step()
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    
    # Evaluation
    print("\nEvaluating on test set...")
    test_accuracy, test_loss = evaluate_model(model, testloader, criterion, device)
    
    # Results summary
    results = {
        'config': config,
        'train_accuracies': train_accuracies,
        'train_losses': train_losses,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'training_time': training_time,
        'model_parameters': count_parameters(model),
        'device': str(device)
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), 'results/model.pth')
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Model Parameters: {count_parameters(model):,}")
    
    # Return metric for optimization (higher is better)
    return test_accuracy


if __name__ == "__main__":
    main() 