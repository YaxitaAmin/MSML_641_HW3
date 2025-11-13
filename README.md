# RNN Sentiment Classification - Comparative Analysis

A comprehensive comparative study of Recurrent Neural Network architectures for binary sentiment classification on the IMDb Movie Review Dataset. This project systematically evaluates 10 different model configurations to identify optimal design choices.

## 📊 Project Overview

This research project implements and compares multiple RNN architectures (Simple RNN, LSTM, Bidirectional LSTM) for sentiment analysis, achieving **81.34% accuracy** through systematic experimentation with various hyperparameters and architectural choices.
**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU, CUDA 12.7

## 🎯 Best Model Performance

- **Architecture:** LSTM (2 layers, 64 hidden units)
- **Test Accuracy:** 81.34%
- **F1 Score:** 81.32%
- **Configuration:** 
  - Activation: ReLU
  - Optimizer: Adam (lr=0.001)
  - Sequence Length: 100 tokens
  - Dropout: 0.4
  - Embedding Dimension: 100
- **Training Time:** 3.61s/epoch

## 📁 Project Structure

```
sentiment-classification/
│
├── data/
│   ├── raw/
│   │   └── IMDB Dataset.csv           # Raw IMDb dataset (50K reviews)
│   └── processed/
│       ├── train_sequences.npy        # Preprocessed training sequences
│       ├── test_sequences.npy         # Preprocessed test sequences
│       └── tokenizer.pkl              # Fitted tokenizer object
│
├── models/
│   ├── rnn.py                         # Simple RNN implementation
│   ├── lstm.py                        # LSTM implementation
│   ├── bilstm.py                      # Bidirectional LSTM implementation
│   └── base_model.py                  # Base model class
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis and statistics
│   ├── 02_preprocessing.ipynb         # Data cleaning and tokenization
│   ├── 03_model_experiments.ipynb     # Model training and evaluation
│   └── 04_results_visualization.ipynb # Performance analysis and plots
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset class and data loading
│   │   └── preprocessing.py           # Text preprocessing pipeline
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── architectures.py           # Model architecture definitions
│   │   └── training.py                # Training loop and evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # Evaluation metrics
│       ├── visualization.py           # Plotting functions
│       └── config.py                  # Configuration parameters
│
├── experiments/
│   ├── experiment_configs/
│   │   ├── exp_01_rnn_baseline.yaml
│   │   ├── exp_02_lstm_baseline.yaml
│   │   ├── exp_03_bilstm.yaml
│   │   └── ...
│   │
│   └── results/
│       ├── metrics/
│       │   └── all_experiments.csv    # Consolidated results
│       ├── checkpoints/
│       │   └── best_model.pth         # Best model weights
│       └── logs/
│           └── training_logs.txt      # Training logs
│
├── results/
│   ├── figures/
│   │   ├── sequence_length_impact.png
│   │   ├── architecture_comparison.png
│   │   ├── optimizer_heatmap.png
│   │   └── training_curves.png
│   │
│   └── tables/
│       └── performance_summary.csv
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_training.py
│
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment file
├── train.py                            # Main training script
├── evaluate.py                         # Model evaluation script
├── predict.py                          # Inference script
├── config.yaml                         # Global configuration
├── README.md                           # This file
└── report.pdf                          # Full comparative analysis report
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rnn-sentiment-classification.git
cd rnn-sentiment-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch==2.0.1
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
pyyaml==6.0
```

### Training

```bash
# Train with best configuration
python train.py --config experiments/experiment_configs/exp_09_lstm_seq100.yaml

# Train with custom parameters
python train.py --model lstm --seq_len 100 --optimizer adam --activation relu --epochs 10
```

### Evaluation

```bash
# Evaluate best model
python evaluate.py --checkpoint experiments/results/checkpoints/best_model.pth

# Run predictions on custom text
python predict.py --text "This movie was absolutely fantastic!" --checkpoint best_model.pth
```

## 📊 Dataset

**Source:** IMDb Movie Review Dataset  
**Total Samples:** 50,000 reviews (balanced)
- Training: 25,000 reviews
- Testing: 25,000 reviews
- Class Distribution: 50% positive, 50% negative

**Preprocessing Pipeline:**
1. Text cleaning (lowercase, remove punctuation)
2. Tokenization (vocabulary size: 10,000)
3. Sequence padding/truncation (tested lengths: 25, 50, 100)
4. Train-test split preservation

**Statistics:**
- Average review length: ~238 tokens
- Vocabulary coverage: 10,000 most frequent words
- OOV handling: Special token for unknown words

## 🔬 Experimental Design

### Models Tested
1. **Simple RNN** - Baseline recurrent architecture
2. **LSTM** - Long Short-Term Memory networks
3. **Bidirectional LSTM** - Forward + backward processing

### Variables Explored
- **Sequence Lengths:** 25, 50, 100 tokens
- **Activation Functions:** ReLU, Tanh, Sigmoid
- **Optimizers:** Adam, SGD, RMSProp
- **Gradient Clipping:** With/without (max norm = 1.0)

### Model Architecture
- **Embedding Layer:** 100 dimensions, vocabulary size 10K
- **Recurrent Layers:** 2 layers, 64 hidden units each
- **Dropout:** 0.4 between layers
- **Output Layer:** Sigmoid activation for binary classification
- **Loss Function:** Binary Cross-Entropy
- **Batch Size:** 32

## 📈 Key Findings

### 1. Sequence Length Impact (Most Critical)
- **25 tokens:** 71.97% accuracy (-4.51%)
- **50 tokens:** 76.48% accuracy (baseline)
- **100 tokens:** 81.34% accuracy (+4.86%)

**Insight:** Longer sequences capture more contextual information. 100 tokens provide sufficient context without excessive computational cost.

### 2. Optimizer Comparison (Critical)
| Optimizer | Accuracy | Status |
|-----------|----------|---------|
| Adam | 76.48% | ✅ Optimal |
| RMSProp | 76.46% | ✅ Good |
| SGD | 50.34% | ❌ Failed |

**Insight:** SGD completely failed to converge (random guessing). Adaptive optimizers (Adam/RMSProp) are essential for RNN training.

### 3. Architecture Performance
| Architecture | Accuracy | Training Time | Cost-Benefit |
|--------------|----------|---------------|--------------|
| RNN | 71.64% | 2.75s/epoch | Low performance |
| LSTM | 76.48% | 2.80s/epoch | ✅ **Optimal** |
| BiLSTM | 76.76% | 3.34s/epoch | Marginal gain (+0.28%) |

**Insight:** LSTM offers best accuracy-to-cost ratio. BiLSTM's 19% increased training time doesn't justify minimal improvement.

### 4. Activation Functions (Minimal Impact)
- **ReLU:** 76.48% (fastest, most efficient)
- **Tanh:** 76.38% (-0.10%)
- **Sigmoid:** 75.82% (-0.66%)

**Insight:** Differences < 1%. Choose ReLU for computational efficiency.

### 5. Gradient Clipping (Negligible Impact)
- **With clipping:** 76.48%
- **Without clipping:** 76.64%
- **Difference:** ±0.16%

**Insight:** Modern LSTMs are inherently stable for this task.

## 🎓 Recommendations

### Optimal Configuration
```python
model_config = {
    'architecture': 'LSTM',
    'activation': 'relu',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'sequence_length': 100,
    'embedding_dim': 100,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.4,
    'batch_size': 32,
    'epochs': 10
}
```

### Best Practices
1. ✅ Always use adaptive optimizers (Adam/RMSProp) for RNN training
2. ✅ Prioritize sequence length optimization over architectural complexity
3. ✅ LSTM is sufficient for sentiment tasks; BiLSTM adds minimal value
4. ✅ Use ReLU activation for speed-accuracy trade-off
5. ✅ Monitor validation metrics to detect overfitting early

### Future Improvements
- Increase hidden size to 128-256 units
- Add attention mechanisms
- Use pre-trained embeddings (GloVe, Word2Vec)
- Implement early stopping
- Try transformer architectures (BERT, RoBERTa) for >90% accuracy

## 🔧 Reproducibility

### Random Seeds
```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### Hardware Requirements
- **GPU:** NVIDIA RTX 3060 or equivalent (6GB+ VRAM recommended)
- **CUDA:** 12.1+
- **RAM:** 8GB+ system memory
- **Storage:** 2GB for dataset and models

### Software Environment
- **Python:** 3.8+
- **PyTorch:** 2.0.1 with CUDA support
- **TensorFlow:** 2.13.0 (for preprocessing only)
- **OS:** Linux/Windows/MacOS

## 📝 Results Summary

| Experiment | Model | Activation | Optimizer | Seq Len | Grad Clip | Accuracy | F1 Score |
|------------|-------|------------|-----------|---------|-----------|----------|----------|
| 1 | RNN | ReLU | Adam | 50 | Yes | 71.64% | 71.49% |
| 2 | LSTM | ReLU | Adam | 50 | Yes | 76.48% | 76.45% |
| 3 | BiLSTM | ReLU | Adam | 50 | Yes | 76.76% | 76.76% |
| 4 | LSTM | Sigmoid | Adam | 50 | Yes | 75.82% | 75.82% |
| 5 | LSTM | Tanh | Adam | 50 | Yes | 76.38% | 76.38% |
| 6 | LSTM | ReLU | SGD | 50 | Yes | 50.34% | 49.76% |
| 7 | LSTM | ReLU | RMSProp | 50 | Yes | 76.46% | 76.44% |
| 8 | LSTM | ReLU | Adam | 25 | Yes | 71.97% | 71.95% |
| **9** | **LSTM** | **ReLU** | **Adam** | **100** | **Yes** | **81.34%** | **81.32%** |
| 10 | LSTM | ReLU | Adam | 50 | No | 76.64% | 76.62% |

## 📄 Documentation

- **Full Report:** See `report.pdf` for comprehensive analysis
- **Notebooks:** Interactive analysis in `notebooks/` directory
- **API Documentation:** Run `pydoc src` for detailed API docs

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📜 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or collaborations, please open an issue or contact:
- **Course:** MSML641
- **Date:** November 12, 2025

**Note:** This is a research project for educational purposes. Results may vary with different random seeds or hardware configurations.
