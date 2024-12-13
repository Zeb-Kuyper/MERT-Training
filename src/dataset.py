import torch
import torchaudio
from torch.utils.data import Dataset
from config import DataConfig
import torchaudio.transforms as T
from datasets import Audio, load_dataset, Dataset
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

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

# Split the dataset into train and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# Function to create .tsv files
def create_tsv(df, split, audio_dir):
    tsv_path = Path(f"{split}.tsv")
    with tsv_path.open("w") as f:
        for _, row in df.iterrows():
            audio_path = Path(audio_dir) / row['sample']
            f.write(f"{audio_path}\t{row['label']}\n")

# Create train.tsv and valid.tsv
create_tsv(train_df, "train", DataConfig.audio_dir)
create_tsv(valid_df, "valid", DataConfig.audio_dir)

# Function to create .km files (dummy implementation)
def create_km(df, split):
    km_path = Path(f"{split}.km")
    with km_path.open("w") as f:
        for _, row in df.iterrows():
            # Dummy frame-aligned pseudo labels
            waveform, sr = torchaudio.load(Path(DataConfig.audio_dir) / row['sample'])
            num_frames = waveform.shape[1] // (sr // 100)  # Assuming 100Hz frame rate
            labels = " ".join(["0"] * num_frames)  # Dummy labels
            f.write(f"{labels}\n")

# Create train.km and valid.km
create_km(train_df, "train")
create_km(valid_df, "valid")

# Create dict.km.txt (dummy dictionary)
with open("dict.km.txt", "w") as f:
    f.write("0 1\n")

if __name__ == "__main__":
    dataset = create_hf_dataset(
        DataConfig.ratings_file,
        DataConfig.audio_dir
    )
