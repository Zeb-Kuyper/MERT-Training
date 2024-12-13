import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor

def train_model(model, train_dataset, val_dataset, config):
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["labels"])
            loss.backward()
            optimizer.step()