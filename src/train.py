from config import DataConfig, ModelConfig
from dataset import AudioDataset
from model import AudioClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from data_utils import prepare_dataset
import pandas as pd

def train_model(model, train_dataset, val_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    device = config.device
    model = model.to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': config.learning_rate},
        {'params': model.base_model.parameters(), 'lr': config.learning_rate * 0.1}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            with torch.cuda.amp.autocast():
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(audio)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                labels = batch["label"].to(device)
                outputs = model(audio)
                val_loss += criterion(outputs, labels).item()
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f"{config.save_path}/best_model.pt")

    return model

if __name__ == "__main__":
    # Example usage
    data_config = DataConfig()
    model_config = ModelConfig()
    train_data = prepare_dataset(data_config)
    model = AudioClassifier(model_config)
    train_dataset = AudioDataset(train_data, model_config)
    val_dataset = AudioDataset(train_data, model_config)
    train_model(model, train_dataset, val_dataset, model_config)