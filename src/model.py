from transformers import AutoModel
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, pretrained_model="m-a-p/MERT-v1-330M", num_classes=2):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        outputs = self.base_model(**x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)