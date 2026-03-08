import torch

class Config:
    """Configuration class for model hyperparameters"""
    
    # Model configurations
    model_name = "vinai/phobert-base-v2"  # PhoBERT-V2 base model
    max_length = 128  # Maximum sequence length
    hidden_size = 768  # PhoBERT hidden size
    num_classes = 2  # Binary classification (positive/negative)
    
    # Training configurations
    batch_size = 24
    num_epochs = 20
    learning_rate = 3e-5
    accumulation_steps = 16
    warmup_ratio = 0.1
    dropout_rate = 0.1
    
    # RCNN configurations
    rcnn_hidden_size = 256
    rcnn_num_layers = 2
    
    # SentiWordNet configurations
    sentiwordnet_dim = 128
    threshold_T = 0.5  # Threshold for pure positive/negative words
    
    # Paths
    data_path = "data/"
    model_save_path = "models/"
    sentiwordnet_path = "Sentiwordnet/vietnamese_sentiwordnet.json"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")