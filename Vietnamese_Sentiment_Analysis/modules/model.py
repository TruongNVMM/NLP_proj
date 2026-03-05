import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence

# Transformers and Vietnamese NLP
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup
)

class RCNN(nn.Module):
    """Recurrent Convolutional Neural Network module"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(RCNN, self).__init__()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size * 2,  # bidirectional
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)
        
        # Transpose for conv1d (batch, channels, seq_len)
        lstm_out = lstm_out.transpose(1, 2)
        
        # Apply convolution
        conv_out = self.conv1d(lstm_out)  # (batch, hidden_size, seq_len)
        conv_out = self.activation(conv_out)
        
        # Global max pooling
        pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
        pooled = pooled.squeeze(2)  # (batch, hidden_size)
        
        return self.dropout(pooled)

class CombViSA(nn.Module):
    """Combined Vietnamese Sentiment Analysis Model"""
    
    def __init__(self, config: Config):
        super(CombViSA, self).__init__()
        self.config = config
        
        # PhoBERT-V2 encoder
        self.phobert = AutoModel.from_pretrained(config.model_name)
        
        # RCNN module for PhoBERT output
        self.rcnn = RCNN(
            input_size=config.hidden_size,
            hidden_size=config.rcnn_hidden_size,
            num_layers=config.rcnn_num_layers,
            dropout=config.dropout_rate
        )
        
        # MLP for PhoBERT features (LMVec)
        self.lm_mlp = nn.Sequential(
            nn.Linear(config.rcnn_hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )
        
        # MLP for SentiWordNet features (SWVec)
        self.sw_mlp = nn.Sequential(
            nn.Linear(config.sentiwordnet_dim * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.num_classes)
        )
        
    def forward(self, input_ids, attention_mask, senti_features):
        # PhoBERT encoding
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # RCNN processing
        rcnn_out = self.rcnn(sequence_output)
        
        # Generate LMVec
        lm_vec = self.lm_mlp(rcnn_out)
        
        # Generate SWVec
        sw_vec = self.sw_mlp(senti_features)
        
        # Concatenate LMVec and SWVec
        combined = torch.cat([lm_vec, sw_vec], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits