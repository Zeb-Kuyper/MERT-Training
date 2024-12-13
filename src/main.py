import torch
from datasets import load_dataset
from model import AudioClassifier
from dataset import AudioDataset
from config import ModelConfig
from train import train_model
import pandas as pd
import os
import torchaudio
from torch.utils.data import Dataset

# Load the CSV data
ratings_df = pd.read_csv('data/raw/MeanCategoryRatingsUSA.csv')

# Create a custom dataset loading function
def load_local_dataset():
    audio_files = []
    for file in os.listdir('audio_samples'):
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_files.append({
                'audio': {
                    'path': os.path.join('audio_samples', file)
                }
            })
    return audio_files

class AudioDataset(Dataset):
    def __getitem__(self, idx):
        audio_path = self.dataset[idx]["audio"]["path"]
        waveform, sample_rate = torchaudio.load(audio_path)
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        return inputs.input_values.squeeze()

def main():
    # Initialize config
    config = ModelConfig()
    
    # Load model
    model = AudioClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes
    )
    
    # Load dataset
    raw_dataset = load_local_dataset()
    
    # Create train/val splits
    train_dataset = AudioDataset(
        raw_dataset[:1000], 
        model.feature_extractor
    )
    val_dataset = AudioDataset(
        raw_dataset[1000:1200], 
        model.feature_extractor
    )
    
    # Train
    train_model(model, train_dataset, val_dataset, config)
    
    # Save model
    torch.save(model.state_dict(), "audio_classifier.pt")

if __name__ == "__main__":
    main()