import torch.nn as nn
from transformers import AutoModel
import torch

class AudioClassifier(nn.Module):
    def __init__(self, base_model, num_classes, freeze_base=True):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get MERT embeddings
        outputs = self.base_model(x)
        # Use mean pooling over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Classification
        return self.classifier(embeddings)