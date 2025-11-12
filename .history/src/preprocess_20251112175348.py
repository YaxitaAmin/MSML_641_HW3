import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import pickle
import os

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_imdb(max_words=10000, max_len=50):
    """
    Load and preprocess IMDb dataset
    
    Args:
        max_words: Maximum number of words to keep
        max_len: Maximum sequence length
    
    Returns:
        X_train, y_train, X_test, y_test, tokenizer
    """
    print("Loading IMDb dataset...")
    (X_train_raw, y_train), (X_test_raw, y_test) = imdb.load_data(num_words=max_words)
    
    # Get the word index
    word_index = imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    # Decode reviews back to text
    def decode_review(encoded_review):
        return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    
    # Decode all reviews
    train_texts = [decode_review(review) for review in X_train_raw]
    test_texts = [decode_review(review) for review in X_test_raw]
    
    # Clean texts
    print("Cleaning texts...")
    train_texts = [clean_text(text) for text in train_texts]
    test_texts = [clean_text(text) for text in test_texts]
    
    # Tokenize
    print("Tokenizing...")
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts)
    
    # Convert to sequences
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)
    
    # Pad sequences
    print(f"Padding sequences to length {max_len}...")
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Vocabulary size: {min(max_words, len(tokenizer.word_index) + 1)}")
    
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