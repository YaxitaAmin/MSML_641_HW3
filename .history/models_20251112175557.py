import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """Basic RNN for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=64, 
                 output_size=1, n_layers=2, dropout=0.3, activation='relu'):
        super(SentimentRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, 
                         batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # RNN
        rnn_out, hidden = self.rnn(embedded)
        
        # Get last output
        out = rnn_out[:, -1, :]
        
        # Apply activation
        out = self.activation(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Fully connected
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

class SentimentLSTM(nn.Module):
    """LSTM for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=64, 
                 output_size=1, n_layers=2, dropout=0.3, activation='relu'):
        super(SentimentLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get last output
        out = lstm_out[:, -1, :]
        
        # Apply activation
        out = self.activation(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Fully connected
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

class SentimentBiLSTM(nn.Module):
    """Bidirectional LSTM for sentiment classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=64, 
                 output_size=1, n_layers=2, dropout=0.3, activation='relu'):
        super(SentimentBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, 
                           batch_first=True, dropout=dropout if n_layers > 1 else 0,
                           bidirectional=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Fully connected layer (note: hidden_size * 2 for bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def _get_activation(self, activation):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # Bidirectional LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get last output
        out = lstm_out[:, -1, :]
        
        # Apply activation
        out = self.activation(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Fully connected
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

def get_model(architecture, vocab_size, embedding_dim=100, hidden_size=64, 
              output_size=1, n_layers=2, dropout=0.3, activation='relu'):
    """
    Factory function to create models
    
    Args:
        architecture: 'rnn', 'lstm', or 'bilstm'
        vocab_size: Size of vocabulary
        embedding_dim: Embedding dimension
        hidden_size: Hidden layer size
        output_size: Output size (1 for binary classification)
        n_layers: Number of recurrent layers
        dropout: Dropout rate
        activation: Activation function ('relu', 'tanh', 'sigmoid')
    
    Returns:
        Model instance
    """
    if architecture.lower() == 'rnn':
        return SentimentRNN(vocab_size, embedding_dim, hidden_size, 
                           output_size, n_layers, dropout, activation)
    elif architecture.lower() == 'lstm':
        return SentimentLSTM(vocab_size, embedding_dim, hidden_size, 
                            output_size, n_layers, dropout, activation)
    elif architecture.lower() == 'bilstm':
        return SentimentBiLSTM(vocab_size, embedding_dim, hidden_size, 
                              output_size, n_layers, dropout, activation)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")