import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import os
from models import get_model
from utils import set_seed, get_device

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set
    
    Returns:
        accuracy, f1_score, predictions, targets
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            
            output = model(data)
            pred = (output >= 0.5).float()
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return accuracy, f1, all_preds, all_targets

def evaluate_saved_model(model_path, config, X_test, y_test, device):
    """Load and evaluate a saved model"""
    
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
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    
    # Create test dataset and loader
    test_dataset = TensorDataset(
        torch.LongTensor(X_test), 
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    accuracy, f1, preds, targets = evaluate_model(model, test_loader, device)
    
    return accuracy, f1, preds, targets

def generate_classification_report(y_true, y_pred, save_path=None):
    """Generate detailed classification report"""
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'])
    print("\nClassification Report:")
    print(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report

def generate_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Neg    Pos")
    print(f"Actual Neg  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos  {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    return cm

def run_full_evaluation(model_name, config, X_test, y_test, device, save_dir='results'):
    """Run complete evaluation and save results"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = f"models/{model_name}.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    # Evaluate
    accuracy, f1, preds, targets = evaluate_saved_model(
        model_path, config, X_test, y_test, device
    )
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score (Macro): {f1:.4f}")
    
    # Generate reports
    report = generate_classification_report(targets, preds, 
                                           f"{save_dir}/{model_name}_report.txt")
    cm = generate_confusion_matrix(targets, preds)
    
    # Save results
    results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'config': config
    }
    
    with open(f"{save_dir}/{model_name}_results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    from preprocess import load_preprocessed_data
    
    # Set random seeds
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Example configuration (should match training config)
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
    
    # Load test data
    _, _, X_test, y_test, _ = load_preprocessed_data(config['seq_length'])
    
    # Model name
    model_name = f"{config['architecture']}_{config['activation']}_{config['optimizer']}_seq{config['seq_length']}_clip{config['grad_clip'] if config['grad_clip'] else 'none'}"
    
    # Run evaluation
    results = run_full_evaluation(model_name, config, X_test, y_test, device)
    
    if results:
        print("\nEvaluation complete!")
        print(f"Results saved to results/{model_name}_results.json")