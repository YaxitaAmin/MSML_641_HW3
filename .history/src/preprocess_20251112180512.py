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

def load_and_preprocess_csv(csv_path='data/IMDB Dataset.csv', max_words=10000, max_len=50):
    """Load and preprocess IMDb dataset from CSV"""
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get text and labels - adjust column names if needed
    texts = df['review'].values  # Change 'review' to your text column name
    labels = df['sentiment'].values  # Change 'sentiment' to your label column name
    
    # Convert labels to binary
    if labels.dtype == 'object':
        labels = np.array([1 if 'positive' in str(label).lower() else 0 for label in labels])
    
    print(f"Positive samples: {np.sum(labels)}")
    print(f"Negative samples: {len(labels) - np.sum(labels)}")
    
    # Clean texts
    print("Cleaning texts...")
    texts = [clean_text(text) for text in texts]
    
    # Split 50/50
    split_idx = len(texts) // 2
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
    
    X_train = tokenizer.texts_to_sequences(X_train_text)
    X_test = tokenizer.texts_to_sequences(X_test_text)
    
    # Pad sequences
    print(f"Padding to length {max_len}...")
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Vocab size: {min(max_words, len(tokenizer.word_index) + 1)}")
    
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
    
    print(f"Saved to {data_dir}/")

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
    np.random.seed(42)
    
    csv_path = 'data/IMDB Dataset.csv'
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        exit(1)
    
    for seq_len in [25, 50, 100]:
        print(f"\n{'='*50}")
        print(f"Processing sequence length: {seq_len}")
        print(f"{'='*50}")
        
        X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_csv(
            csv_path=csv_path,
            max_words=10000, 
            max_len=seq_len
        )
        
        save_preprocessed_data(X_train, y_train, X_test, y_test, tokenizer, seq_len)
    
    print("\n✓ Preprocessing complete!")