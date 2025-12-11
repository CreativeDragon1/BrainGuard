"""
Training script for Alzheimer's MRI Detection Model
Intermediate-level project: Compare multiple architectures with validation
"""

import os
import io
import json
from pathlib import Path
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from models.cnn_model import AlzheimersCNN, ResNetModel, SimpleResNet

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class ParquetMRIDataset(Dataset):
    """Dataset that reads MRI images stored as bytes in parquet."""

    def __init__(self, records, labels, train=True):
        self.records = records  # list of bytes
        self.labels = labels    # list of ints
        self.train = train
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        self.aug = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        raw = self.records[idx]
        img = Image.open(io.BytesIO(raw)).convert('L')
        if self.train:
            img = self.aug(img)
        img = self.base_transform(img)
        return img, self.labels[idx]

class Trainer:
    """Model trainer with validation and model selection"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None, 
                 epochs=50, lr=1e-3, model_name='model'):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.model_name = model_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validating')
            for images, labels in progress_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def test(self):
        """Test model"""
        if self.test_loader is None:
            return None
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc='Testing')
            for images, labels in progress_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
    
    def train(self):
        """Full training loop"""
        print(f"Training {self.model_name} on {DEVICE}")
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best')
                print(f"Best model saved! Accuracy: {val_acc:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
        
        return self.history
    
    def save_checkpoint(self, name='model'):
        """Save model checkpoint"""
        os.makedirs('models', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        if name == 'best':
            path = f'models/best_{self.model_name}.pth'
        else:
            path = f'models/{self.model_name}_{name}.pth'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")

def compute_metrics(y_true, y_pred, class_names=None):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if class_names:
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            class_mask = np.array(y_true) == i
            if class_mask.sum() > 0:
                metrics['per_class'][class_name] = {
                    'precision': precision_score(np.array(y_true)[class_mask], 
                                                np.array(y_pred)[class_mask], 
                                                average='weighted', zero_division=0),
                    'recall': recall_score(np.array(y_true)[class_mask], 
                                          np.array(y_pred)[class_mask], 
                                          average='weighted', zero_division=0),
                }
    
    return metrics

def generate_report(results, model_name, class_names):
    """Generate evaluation report"""
    report = {
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'metrics': results,
        'class_names': class_names
    }
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Alzheimer MRI Detection Model')
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['cnn', 'resnet', 'simple_resnet', 'all'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    set_seed(SEED)
    
    print("="*50)
    print("Alzheimer's MRI Detection Model Training")
    print("="*50)
    print(f"Device: {DEVICE}")
    print(f"Model: {args.model}")
    print("="*50)

    # Load parquet dataset
    train_path = Path('Assets/Datasets/MRI Dataset/train.parquet')
    test_path = Path('Assets/Datasets/MRI Dataset/test.parquet')
    if not train_path.exists():
        raise SystemExit("train.parquet not found. Please place the dataset in Assets/Datasets/MRI Dataset/")

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path) if test_path.exists() else None

    # Extract bytes and labels
    train_records = [row['image']['bytes'] for _, row in df_train.iterrows()]
    train_labels = df_train['label'].tolist()

    # Train/val split
    indices = np.arange(len(train_records))
    np.random.shuffle(indices)
    val_split = int(0.1 * len(indices))
    val_idx = indices[:val_split]
    train_idx = indices[val_split:]

    def subset(records, labels, idxs):
        return [records[i] for i in idxs], [labels[i] for i in idxs]

    tr_recs, tr_labs = subset(train_records, train_labels, train_idx)
    val_recs, val_labs = subset(train_records, train_labels, val_idx)

    train_ds = ParquetMRIDataset(tr_recs, tr_labs, train=True)
    val_ds = ParquetMRIDataset(val_recs, val_labs, train=False)
    test_ds = None
    if df_test is not None:
        ts_recs = [row['image']['bytes'] for _, row in df_test.iterrows()]
        ts_labs = df_test['label'].tolist()
        test_ds = ParquetMRIDataset(ts_recs, ts_labs, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2) if test_ds else None

    # Select model
    if args.model == 'resnet':
        model = ResNetModel(pretrained=True, num_classes=4)
        model_name = 'resnet'
    elif args.model == 'simple_resnet':
        model = SimpleResNet(pretrained=True, num_classes=4)
        model_name = 'simple_resnet'
    else:
        model = AlzheimersCNN(num_classes=4)
        model_name = 'cnn'

    trainer = Trainer(model, train_loader, val_loader, test_loader, epochs=args.epochs, lr=args.lr, model_name=model_name)
    history = trainer.train()

    if test_loader:
        test_results = trainer.test()
        print("Test accuracy:", test_results['accuracy'])

    # Save final checkpoint
    trainer.save_checkpoint('final')
