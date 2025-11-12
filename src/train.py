import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import json
from models import get_model
from utils import set_seed, get_device
import sys

def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping if specified
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = (output >= 0.5).float()
        correct += (pred == target).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = (output >= 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_preds, all_targets

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs=10, grad_clip=None, save_path=None):
    """
    Train the model
    
    Returns:
        history: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }
    
    best_val_acc = 0
    
    print("\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time (s)':<10}")
    print("-" * 76)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, grad_clip)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {epoch_time:<10.2f}")
        
        # Save best model
        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
    
    return history

def run_experiment(config, X_train, y_train, X_test, y_test, device):
    """Run a single experiment with given configuration"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.LongTensor(X_train), 
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(X_test), 
        torch.LongTensor(y_test)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = get_model(
        architecture=config['architecture'],
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        activation=config['activation']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    
    if config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    # Gradient clipping value
    grad_clip = config.get('grad_clip', None)
    
    # Create save directory
    os.makedirs('models', exist_ok=True)
    model_name = f"{config['architecture']}_{config['activation']}_{config['optimizer']}_seq{config['seq_length']}_clip{grad_clip if grad_clip else 'none'}"
    save_path = f"models/{model_name}.pt"
    
    # Train
    print(f"\n{'='*76}")
    print(f"Training: {model_name}")
    print(f"{'='*76}")
    
    history = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        device, epochs=config['epochs'], grad_clip=grad_clip, 
        save_path=save_path
    )
    
    # Calculate average epoch time
    avg_epoch_time = np.mean(history['epoch_time'])
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(save_path))
    
    return history, avg_epoch_time, model_name

if __name__ == "__main__":
    from preprocess import load_preprocessed_data
    
    # Set random seeds
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Example configuration
    config = {
        'architecture': 'lstm',
        'activation': 'relu',
        'optimizer': 'adam',
        'seq_length': 50,
        'vocab_size': 10000,
        'embedding_dim': 100,
        'hidden_size': 64,
        'n_layers': 2,
        'dropout': 0.4,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 10,
        'grad_clip': 1.0
    }
    
    # Load data
    X_train, y_train, X_test, y_test, tokenizer = load_preprocessed_data(config['seq_length'])
    
    # Run experiment
    history, avg_epoch_time, model_name = run_experiment(
        config, X_train, y_train, X_test, y_test, device
    )
    
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")