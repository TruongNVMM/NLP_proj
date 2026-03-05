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

class VSMECDataset(Dataset):
    """VSMEC Sentiment Analysis Dataset"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 sentiwordnet: VietnameseSentiWordNet, config: Config):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentiwordnet = sentiwordnet
        self.config = config
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # PhoBERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Extract SentiWordNet features
        pos_vec, neg_vec = self.sentiwordnet.extract_sentiment_vectors(text)
        senti_features = np.concatenate([pos_vec, neg_vec])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'senti_features': torch.FloatTensor(senti_features),
            'label': torch.LongTensor([label])
        }

class Trainer:
    """Training and evaluation manager"""
    
    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.device = config.device
        
    def train(self, train_loader, val_loader):
        """Train the model"""
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_f1 = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                senti_features = batch['senti_features'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, attention_mask, senti_features)
                loss = criterion(logits, labels)
                
                # Gradient accumulation
                loss = loss / self.config.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.accumulation_steps
                
                # Store predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate training metrics
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                train_labels, train_preds, average='weighted'
            )
            
            # Validation phase
            val_precision, val_recall, val_f1 = self.evaluate(val_loader)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"Train - Loss: {train_loss/len(train_loader):.4f}, "
                  f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config.model_save_path, 'best_model.pt'))
                print(f"Best model saved with F1: {best_f1:.4f}")
            
            print("-" * 50)
    
    def evaluate(self, data_loader):
        """Evaluate the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                senti_features = batch['senti_features'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                logits = self.model(input_ids, attention_mask, senti_features)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return precision, recall, f1

def preprocess_text(text: str) -> str:
    """Preprocess Vietnamese text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    return text.strip()

class SentimentPredictor:
    """Sentiment prediction interface"""
    
    def __init__(self, model_path: str, config: Config):
        self.config = config
        self.device = config.device
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.sentiwordnet = VietnameseSentiWordNet(config)
        self.sentiwordnet.expand_sentiwordnet()
        
        # Load model
        self.model = CombViSA(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        # Preprocess
        text = preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        # Extract SentiWordNet features
        pos_vec, neg_vec = self.sentiwordnet.extract_sentiment_vectors(text)
        senti_features = torch.FloatTensor(np.concatenate([pos_vec, neg_vec])).unsqueeze(0)
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        senti_features = senti_features.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, senti_features)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
        
        return {
            'text': text,
            'prediction': 'Positive' if pred == 1 else 'Negative',
            'confidence': probs[0, pred].item(),
            'probabilities': {
                'negative': probs[0, 0].item(),
                'positive': probs[0, 1].item()
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict sentiment for multiple texts"""
        return [self.predict(text) for text in texts]