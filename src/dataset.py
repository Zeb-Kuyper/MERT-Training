import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, raw_dataset, feature_extractor):
        self.dataset = raw_dataset
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        audio = self.dataset[idx]["audio"]["array"]
        # Use feature extractor's native sampling rate
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt"
        )
        return inputs.input_values.squeeze()