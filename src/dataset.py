import torch
import torchaudio
from torch.utils.data import Dataset
from config import DataConfig
import torchaudio.transforms as T
from datasets import Audio, load_dataset, Dataset
import pandas as pd
import os
from pathlib import Path

class AudioDataset(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.config = config
        
    def _process_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)
        
        # Pad/trim to target length
        if waveform.shape[1] < self.config.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.config.target_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.config.target_length]
            
        return waveform

    def __getitem__(self, idx):
        audio_path = self.data[idx]["audio_path"]
        label = self.data[idx]["label"]
        
        waveform = self._process_audio(audio_path)
        return {"audio": waveform, "label": label}

    def __len__(self):
        return len(self.data)

# Load and process your data
def create_hf_dataset(data_path, audio_dir):
    # Read CSV file 
    df = pd.read_csv(data_path)
    
    # Create dataset dict
    dataset_dict = {
        "audio": [f"{audio_dir}/{file}" for file in df['sample']],
        "label": df.iloc[:, 1:].values.tolist()
    }
    
    # Convert to HF Dataset
    dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio())
    
    return dataset

if __name__ == "__main__":
    dataset = create_hf_dataset(
        DataConfig.ratings_file,
        DataConfig.audio_dir
    )
