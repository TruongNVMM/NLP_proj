import config 
import pandas as pd
import os

def load_VSMEC_data(config):
    """Load and preprocess VSMEC dataset from excel files"""
    print("Loading VSMEC dataset...")
    
    emotion_dict = {
        'Other': 0,
        'Disgust': 1,
        'Enjoyment': 2,
        'Anger': 3,
        'Surprise': 4,
        'Sadness': 5,
        'Fear': 6
    }
    
    data_dir = "data"
    train_df = pd.read_excel(os.path.join(data_dir, 'train_nor_811.xlsx')).dropna(subset=['Sentence', 'Emotion'])
    val_df = pd.read_excel(os.path.join(data_dir, 'valid_nor_811.xlsx')).dropna(subset=['Sentence', 'Emotion'])
    test_df = pd.read_excel(os.path.join(data_dir, 'test_nor_811.xlsx')).dropna(subset=['Sentence', 'Emotion'])
    
    train_texts = train_df['Sentence'].astype(str).tolist()
    train_labels = train_df['Emotion'].map(emotion_dict).astype(int).tolist()
    
    val_texts = val_df['Sentence'].astype(str).tolist()
    val_labels = val_df['Emotion'].map(emotion_dict).astype(int).tolist()
    
    test_texts = test_df['Sentence'].astype(str).tolist()
    test_labels = test_df['Emotion'].map(emotion_dict).astype(int).tolist()
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def main():
    # Initialize configuration
    config = Config()
    print(f"Using device: {config.device}")
    
    # Create directories
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # Initialize tokenizer
    print("Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Initialize SentiWordNet
    print("Initializing Vietnamese SentiWordNet...")
    sentiwordnet = VietnameseSentiWordNet(config)
    sentiwordnet.expand_sentiwordnet()
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_VSMEC_data(config)
    
    # Preprocess texts
    train_texts = [preprocess_text(text) for text in train_texts]
    val_texts = [preprocess_text(text) for text in val_texts]
    test_texts = [preprocess_text(text) for text in test_texts]
    
    # Split train into train/val
    
    print(f"Train size: {len(train_texts)}")
    print(f"Validation size: {len(val_texts)}")
    print(f"Test size: {len(test_texts)}")
    
    # Create datasets
    train_dataset = VSMECDataset(train_texts, train_labels, tokenizer, sentiwordnet, config)
    val_dataset = VSMECDataset(val_texts, val_labels, tokenizer, sentiwordnet, config)
    test_dataset = VSMECDataset(test_texts, test_labels, tokenizer, sentiwordnet, config)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing CombViSA model...")
    model = CombViSA(config)
    model.to(config.device)
    
    # Initialize trainer
    trainer = Trainer(model, config)
    
    # Train model
    print("\nStarting training...")
    trainer.train(train_loader, val_loader)
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(config.model_save_path, 'best_model.pt')))
    test_precision, test_recall, test_f1 = trainer.evaluate(test_loader)
    
    print("\nFinal Test Results:")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-Score: {test_f1:.4f}")

if __name__ == "__main__":
    main()
