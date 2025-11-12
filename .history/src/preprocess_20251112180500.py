import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

def clean_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_imdb_csv(csv_path='E:\MSML641_HW3\data\IMDB Dataset.csv', max_words=10000, max_len=50, 
                                  text_column='review', label_column='sentiment',
                                  test_split=0.5):
    """
    Load and preprocess IMDb dataset from CSV file
    
    Args:
        csv_path: Path to CSV file
        max_words: Maximum number of words to keep
        max_len: Maximum sequence length
        text_column: Name of column containing text
        label_column: Name of column containing labels
        test_split: Fraction of data to use for testing (default 0.5 for 50/50 split)
    
    Returns:
        X_train, y_train, X_test, y_test, tokenizer
    """
    print(f"Loading IMDb dataset from {csv_path}...")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total samples loaded: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if required columns exist
    if text_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Text column '{text_column}' not found in CSV")
    
    if label_column not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Label column '{label_column}' not found in CSV")
    
    # Get texts and labels
    texts = df[text_column].values
    labels = df[label_column].values
    
    # Convert labels to binary (0 or 1)
    # Common formats: 'positive'/'negative', 'pos'/'neg', 1/0
    if labels.dtype == 'object':
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")
        
        # Map to binary
        if 'positive' in str(unique_labels[0]).lower():
            labels = np.array([1 if 'positive' in str(label).lower() else 0 for label in labels])
        elif 'pos' in str(unique_labels[0]).lower():
            labels = np.array([1 if 'pos' in str(label).lower() else 0 for label in labels])
        else:
            # Try to convert directly
            labels = np.array([int(label) for label in labels])
    
    print(f"Label distribution: {np.bincount(labels)}")
    
    # Clean texts
    print("Cleaning texts...")
    texts = [clean_text(text) for text in texts]
    
    # Split into train and test (50/50 as per requirements)
    split_idx = int(len(texts) * test_split)
    
    # Use first half for training, second half for testing
    X_train_text = texts[:split_idx]
    y_train = labels[:split_idx]
    X_test_text = texts[split_idx:]
    y_test = labels[split_idx:]
    
    print(f"Training samples: {len(X_train_text)}")
    print(f"Test samples: {len(X_test_text)}")
    
    # Tokenize
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train_text)
    
    # Convert to sequences
    X_train = tokenizer.texts_to_sequences(X_train_text)
    X_test = tokenizer.texts_to_sequences(X_test_text)
    
    # Pad sequences
    print(f"Padding sequences to length {max_len}...")
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
    
    vocab_size = min(max_words, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")
    
    # Print some statistics
    avg_train_len = np.mean([len(seq) for seq in tokenizer.texts_to_sequences(X_train_text)])
    avg_test_len = np.mean([len(seq) for seq in tokenizer.texts_to_sequences(X_test_text)])
    print(f"Average review length (train): {avg_train_len:.2f} words")
    print(f"Average review length (test): {avg_test_len:.2f} words")
    
    return X_train, y_train, X_test, y_test, tokenizer

def save_preprocessed_data(X_train, y_train, X_test, y_test, tokenizer, seq_len, data_dir='data'):
    """Save preprocessed data"""
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(f'{data_dir}/X_train_{seq_len}.npy', X_train)
    np.save(f'{data_dir}/y_train_{seq_len}.npy', y_train)
    np.save(f'{data_dir}/X_test_{seq_len}.npy', X_test)
    np.save(f'{data_dir}/y_test_{seq_len}.npy', y_test)
    
    with open(f'{data_dir}/tokenizer_{seq_len}.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Data saved to {data_dir}/")

def load_preprocessed_data(seq_len, data_dir='data'):
    """Load preprocessed data"""
    X_train = np.load(f'{data_dir}/X_train_{seq_len}.npy')
    y_train = np.load(f'{data_dir}/y_train_{seq_len}.npy')
    X_test = np.load(f'{data_dir}/X_test_{seq_len}.npy')
    y_test = np.load(f'{data_dir}/y_test_{seq_len}.npy')
    
    with open(f'{data_dir}/tokenizer_{seq_len}.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    return X_train, y_train, X_test, y_test, tokenizer

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    
    # Preprocess for different sequence lengths
    for seq_len in [25, 50, 100]:
        print(f"\n{'='*50}")
        print(f"Processing sequence length: {seq_len}")
        print(f"{'='*50}")
        
        X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(
            max_words=10000, 
            max_len=seq_len
        )
        
        save_preprocessed_data(X_train, y_train, X_test, y_test, tokenizer, seq_len)
    
    print("\nPreprocessing complete!")