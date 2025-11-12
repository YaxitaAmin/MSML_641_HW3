import os
import json
import numpy as np
import pandas as pd
from itertools import product
from preprocess import load_preprocessed_data
from train import run_experiment
from evaluate import evaluate_saved_model
from utils import (set_seed, get_device, save_results_to_csv, 
                   print_summary_table, get_best_model,
                   plot_sequence_length_comparison, plot_architecture_comparison,
                   plot_training_history, plot_heatmap)

def generate_experiment_configs():
    """Generate all experiment configurations"""
    
    # Define variations
    architectures = ['rnn', 'lstm', 'bilstm']
    activations = ['sigmoid', 'relu', 'tanh']
    optimizers = ['adam', 'sgd', 'rmsprop']
    seq_lengths = [25, 50, 100]
    grad_clips = [None, 1.0]  # No clipping vs clipping
    
    # Base configuration
    base_config = {
        'vocab_size': 10000,
        'embedding_dim': 100,
        'hidden_size': 64,
        'n_layers': 2,
        'dropout': 0.4,
        'batch_size': 32,
        'lr': 0.001,
        'epochs': 10
    }
    
    # Generate all combinations
    configs = []
    
    # Systematic testing: vary one factor at a time
    # Keep seq_length=50, optimizer=adam, activation=relu as baseline
    
    # 1. Test different architectures
    for arch in architectures:
        config = base_config.copy()
        config.update({
            'architecture': arch,
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 50,
            'grad_clip': 1.0
        })
        configs.append(config)
    
    # 2. Test different activations (using LSTM as baseline)
    for act in activations:
        config = base_config.copy()
        config.update({
            'architecture': 'lstm',
            'activation': act,
            'optimizer': 'adam',
            'seq_length': 50,
            'grad_clip': 1.0
        })
        configs.append(config)
    
    # 3. Test different optimizers (using LSTM as baseline)
    for opt in optimizers:
        config = base_config.copy()
        config.update({
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': opt,
            'seq_length': 50,
            'grad_clip': 1.0
        })
        configs.append(config)
    
    # 4. Test different sequence lengths (using LSTM as baseline)
    for seq_len in seq_lengths:
        config = base_config.copy()
        config.update({
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': seq_len,
            'grad_clip': 1.0
        })
        configs.append(config)
    
    # 5. Test gradient clipping effect (using LSTM as baseline)
    for grad_clip in grad_clips:
        config = base_config.copy()
        config.update({
            'architecture': 'lstm',
            'activation': 'relu',
            'optimizer': 'adam',
            'seq_length': 50,
            'grad_clip': grad_clip
        })
        configs.append(config)
    
    # Remove duplicates based on key parameters
    unique_configs = []
    seen = set()
    for config in configs:
        key = (config['architecture'], config['activation'], 
               config['optimizer'], config['seq_length'], config['grad_clip'])
        if key not in seen:
            seen.add(key)
            unique_configs.append(config)
    
    return unique_configs

def run_all_experiments():
    """Run all experiments and save results"""
    
    # Set seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Generate configs
    configs = generate_experiment_configs()
    
    print(f"\nTotal experiments to run: {len(configs)}")
    print("="*80)
    
    # Results storage
    all_results = []
    all_histories = {}
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Run each experiment
    for i, config in enumerate(configs):
        print(f"\n\n{'#'*80}")
        print(f"EXPERIMENT {i+1}/{len(configs)}")
        print(f"{'#'*80}")
        
        # Load data for this sequence length
        X_train, y_train, X_test, y_test, tokenizer = load_preprocessed_data(config['seq_length'])
        
        try:
            # Train model
            history, avg_epoch_time, model_name = run_experiment(
                config, X_train, y_train, X_test, y_test, device
            )
            
            # Evaluate model
            model_path = f"models/{model_name}.pt"
            accuracy, f1, _, _ = evaluate_saved_model(
                model_path, config, X_test, y_test, device
            )
            
            # Store results
            result = {
                'model_name': model_name,
                'architecture': config['architecture'],
                'activation': config['activation'],
                'optimizer': config['optimizer'],
                'seq_length': config['seq_length'],
                'grad_clip': config['grad_clip'],
                'accuracy': accuracy,
                'f1_score': f1,
                'avg_epoch_time': avg_epoch_time,
                'best_val_acc': max(history['val_acc']),
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            }
            
            all_results.append(result)
            all_histories[model_name] = history
            
            print(f"\nTest Accuracy: {accuracy:.4f}")
            print(f"Test F1 Score: {f1:.4f}")
            print(f"Avg Epoch Time: {avg_epoch_time:.2f}s")
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            continue
    
    # Save all results
    results_df = save_results_to_csv(all_results, 'results/metrics.csv')
    
    # Save histories
    with open('results/training_histories.json', 'w') as f:
        json.dump(all_histories, f, indent=4)
    
    print("\n\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    
    # Print summary
    print_summary_table(results_df)
    
    # Get best models
    print("\n")
    best_acc_model = get_best_model(results_df, 'accuracy')
    print("\n")
    best_f1_model = get_best_model(results_df, 'f1_score')
    
    # Generate plots
    print("\n\nGenerating plots...")
    
    # Sequence length comparison
    plot_sequence_length_comparison(results_df, 'results/plots/seq_length_comparison.png')
    
    # Architecture comparison
    plot_architecture_comparison(results_df, 'results/plots/architecture_comparison.png')
    
    # Training history for best and worst models
    best_model_name = best_acc_model['model_name']
    worst_idx = results_df['accuracy'].idxmin()
    worst_model_name = results_df.loc[worst_idx, 'model_name']
    
    if best_model_name in all_histories:
        plot_training_history(all_histories[best_model_name], 
                            f'results/plots/best_model_history.png')
    
    if worst_model_name in all_histories:
        plot_training_history(all_histories[worst_model_name], 
                            f'results/plots/worst_model_history.png')
    
    # Heatmap: Activation vs Optimizer
    lstm_df = results_df[results_df['architecture'] == 'lstm']
    if len(lstm_df) > 0:
        plot_heatmap(lstm_df, 'optimizer', 'activation', 'accuracy',
                    'results/plots/activation_optimizer_heatmap.png')
    
    print("\nAll plots saved to results/plots/")
    print("\n" + "="*80)
    print("EXPERIMENT SUITE COMPLETE!")
    print("="*80)
    
    return results_df, all_histories

if __name__ == "__main__":
    results_df, histories = run_all_experiments()