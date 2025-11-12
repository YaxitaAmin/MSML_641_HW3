import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device (CPU or GPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def plot_training_history(history, save_path=None):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_comparison_metrics(results_df, metric='accuracy', group_by='architecture', save_path=None):
    """Plot comparison of metrics across different configurations"""
    plt.figure(figsize=(10, 6))
    
    if group_by in results_df.columns:
        results_df.groupby(group_by)[metric].plot(kind='bar', legend=True)
    else:
        results_df[metric].plot(kind='bar')
    
    plt.xlabel(group_by.capitalize())
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} Comparison by {group_by.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_sequence_length_comparison(results_df, save_path=None):
    """Plot metrics vs sequence length"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by sequence length
    seq_groups = results_df.groupby('seq_length')
    
    # Plot accuracy
    seq_lengths = []
    accuracies = []
    f1_scores = []
    
    for seq_len, group in seq_groups:
        seq_lengths.append(seq_len)
        accuracies.append(group['accuracy'].mean())
        f1_scores.append(group['f1_score'].mean())
    
    axes[0].plot(seq_lengths, accuracies, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Sequence Length')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(seq_lengths, f1_scores, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_architecture_comparison(results_df, save_path=None):
    """Plot comparison across architectures"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    arch_groups = results_df.groupby('architecture')
    
    architectures = []
    accuracies = []
    f1_scores = []
    
    for arch, group in arch_groups:
        architectures.append(arch.upper())
        accuracies.append(group['accuracy'].mean())
        f1_scores.append(group['f1_score'].mean())
    
    x = np.arange(len(architectures))
    width = 0.35
    
    axes[0].bar(x, accuracies, width, label='Accuracy', alpha=0.8)
    axes[0].set_xlabel('Architecture')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Architecture')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(architectures)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x, f1_scores, width, label='F1 Score', alpha=0.8, color='orange')
    axes[1].set_xlabel('Architecture')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score by Architecture')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(architectures)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_heatmap(results_df, x_col, y_col, value_col, save_path=None):
    """Plot heatmap of results"""
    pivot_table = results_df.pivot_table(values=value_col, index=y_col, columns=x_col)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': value_col.capitalize()})
    plt.title(f'{value_col.capitalize()} Heatmap: {y_col.capitalize()} vs {x_col.capitalize()}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def save_results_to_csv(results_list, save_path='results/metrics.csv'):
    """Save all results to CSV file"""
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")
    return df

def load_results_from_csv(csv_path='results/metrics.csv'):
    """Load results from CSV file"""
    return pd.read_csv(csv_path)

def print_summary_table(results_df):
    """Print formatted summary table"""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    # Select and order columns
    columns = ['architecture', 'activation', 'optimizer', 'seq_length', 
               'grad_clip', 'accuracy', 'f1_score', 'avg_epoch_time']
    
    display_df = results_df[columns].copy()
    display_df['grad_clip'] = display_df['grad_clip'].fillna('No')
    display_df['grad_clip'] = display_df['grad_clip'].apply(lambda x: 'Yes' if x != 'No' else 'No')
    
    # Format numbers
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4f}")
    display_df['f1_score'] = display_df['f1_score'].apply(lambda x: f"{x:.4f}")
    display_df['avg_epoch_time'] = display_df['avg_epoch_time'].apply(lambda x: f"{x:.2f}")
    
    # Rename columns for display
    display_df.columns = ['Model', 'Activation', 'Optimizer', 'Seq Length', 
                         'Grad Clip', 'Accuracy', 'F1', 'Epoch Time (s)']
    
    print(display_df.to_string(index=False))
    print("="*100)

def get_best_model(results_df, metric='accuracy'):
    """Get best performing model based on metric"""
    best_idx = results_df[metric].idxmax()
    best_model = results_df.loc[best_idx]
    
    print(f"\nBest model by {metric}:")
    print(f"Architecture: {best_model['architecture']}")
    print(f"Activation: {best_model['activation']}")
    print(f"Optimizer: {best_model['optimizer']}")
    print(f"Sequence Length: {best_model['seq_length']}")
    print(f"Gradient Clipping: {best_model.get('grad_clip', 'No')}")
    print(f"{metric.capitalize()}: {best_model[metric]:.4f}")
    
    return best_model