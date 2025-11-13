import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import json
import os
import sys
from models import get_model
from utils import set_seed, get_device

def find_data_directory():
    """Find where the preprocessed data is located"""
    # Check multiple possible locations
    possible_dirs = [
        'data',           # Current directory/data
        '../data',        # Parent directory/data
        '.',              # Current directory
        '..',             # Parent directory
    ]
    
    for data_dir in possible_dirs:
        # Check if at least one preprocessed file exists
        if os.path.exists(os.path.join(data_dir, 'X_train_50.npy')):
            return data_dir
    
    return None

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
    f1 = f1_score(all_targets, all_preds, average='binary')
    
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")
    
    return report

def generate_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Neg    Pos")
    print(f"Actual Neg  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       Pos  {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Specificity:     {specificity:.4f}")
    
    return cm

def run_full_evaluation(model_name, config, X_test, y_test, device, save_dir='results'):
    """Run complete evaluation and save results"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check multiple possible model locations
    possible_paths = [
        f"models/{model_name}.pt",
        f"../models/{model_name}.pt",
        f"experiments/results/checkpoints/{model_name}.pt"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print(f"Error: Model not found!")
        print(f"Searched locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nAvailable models:")
        for check_dir in ['models', '../models', 'experiments/results/checkpoints']:
            if os.path.exists(check_dir):
                print(f"\nIn {check_dir}:")
                for f in os.listdir(check_dir):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        print(f"  - {f}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Evaluate
        accuracy, f1, preds, targets = evaluate_saved_model(
            model_path, config, X_test, y_test, device
        )
        
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Test F1 Score: {f1:.4f}")
        
        # Generate reports
        report = generate_classification_report(
            targets, preds, 
            f"{save_dir}/{model_name}_report.txt"
        )
        cm = generate_confusion_matrix(targets, preds)
        
        # Save results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'config': config
        }
        
        results_path = f"{save_dir}/{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_path}")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_preprocessed_data_auto(seq_len):
    """Load preprocessed data with automatic directory detection"""
    data_dir = find_data_directory()
    
    if data_dir is None:
        raise FileNotFoundError(f"Could not find preprocessed data files for sequence length {seq_len}")
    
    print(f"Loading data from: {data_dir}")
    
    import pickle
    
    X_train = np.load(os.path.join(data_dir, f'X_train_{seq_len}.npy'))
    y_train = np.load(os.path.join(data_dir, f'y_train_{seq_len}.npy'))
    X_test = np.load(os.path.join(data_dir, f'X_test_{seq_len}.npy'))
    y_test = np.load(os.path.join(data_dir, f'y_test_{seq_len}.npy'))
    
    with open(os.path.join(data_dir, f'tokenizer_{seq_len}.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    
    return X_train, y_train, X_test, y_test, tokenizer

def evaluate_all_models(device, save_dir='results'):
    """Evaluate all saved models"""
    
    # Find models directory
    models_dirs = ['models', '../models', 'experiments/results/checkpoints']
    models_dir = None
    
    for md in models_dirs:
        if os.path.exists(md):
            models_dir = md
            break
    
    if models_dir is None:
        print("Error: 'models' directory not found!")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.pth')]
    
    if not model_files:
        print(f"No model files found in '{models_dir}' directory!")
        return
    
    print(f"Found {len(model_files)} models to evaluate in {models_dir}")
    
    results_list = []
    
    for model_file in model_files:
        model_name = model_file.replace('.pt', '').replace('.pth', '')
        
        # Parse model name to extract config
        # Expected format: architecture_activation_optimizer_seqXX_clipX.X
        parts = model_name.split('_')
        
        if len(parts) < 5:
            print(f"Skipping {model_name} - invalid name format")
            continue
        
        try:
            architecture = parts[0]
            activation = parts[1]
            optimizer = parts[2]
            seq_length = int(parts[3].replace('seq', ''))
            grad_clip = parts[4].replace('clip', '')
            grad_clip = None if grad_clip == 'none' else float(grad_clip)
            
            config = {
                'architecture': architecture,
                'activation': activation,
                'optimizer': optimizer,
                'seq_length': seq_length,
                'vocab_size': 10000,
                'embedding_dim': 100,
                'hidden_size': 64,
                'n_layers': 2,
                'dropout': 0.4,
                'batch_size': 32,
                'lr': 0.001,
                'epochs': 10,
                'grad_clip': grad_clip
            }
            
            # Load data for this sequence length
            try:
                X_train, y_train, X_test, y_test, tokenizer = load_preprocessed_data_auto(seq_length)
            except FileNotFoundError as e:
                print(f"Skipping {model_name}: {e}")
                continue
            
            result = run_full_evaluation(model_name, config, X_test, y_test, device, save_dir)
            
            if result:
                results_list.append(result)
                
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    if results_list:
        summary_path = f"{save_dir}/evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results_list, f, indent=4)
        print(f"\nEvaluation summary saved to {summary_path}")
        
        # Print summary table
        print("\n" + "="*100)
        print("EVALUATION SUMMARY")
        print("="*100)
        print(f"{'Model Name':<50} {'Accuracy':<12} {'F1 Score':<12}")
        print("-"*100)
        for result in sorted(results_list, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['model_name']:<50} {result['accuracy']:<12.4f} {result['f1_score']:<12.4f}")
        print("="*100)
    else:
        print("\nNo models were successfully evaluated!")

if __name__ == "__main__":
    # Set random seeds
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Check if we should evaluate all models or a specific one
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Evaluate all models
        print("\nEvaluating all models...")
        evaluate_all_models(device, save_dir='results')
    else:
        # Evaluate single model (best model from your report)
        config = {
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 100,
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
        try:
            X_train, y_train, X_test, y_test, tokenizer = load_preprocessed_data_auto(config['seq_length'])
            
            # Model name
            model_name = f"{config['architecture']}_{config['activation']}_{config['optimizer']}_seq{config['seq_length']}_clip{config['grad_clip'] if config['grad_clip'] else 'none'}"
            
            # Run evaluation
            results = run_full_evaluation(model_name, config, X_test, y_test, device)
            
            if results:
                print("\n✓ Evaluation complete!")
                print(f"Results saved to results/{model_name}_results.json")
            else:
                print("\n✗ Evaluation failed!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
