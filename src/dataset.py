import torch
import torchaudio
from torch.utils.data import Dataset
from config import DataConfig
import torchaudio.transforms as T
import datasets
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
def resample_audio(audio_path, target_sr=24000):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    return waveform

def create_hf_dataset(data_path, audio_dir):
    # Read CSV file 
    emotion_cols = ['amusing', 'angry', 'annoying', 'anxious/tense', 
                   'awe-inspiring/amazing', 'beautiful', 'bittersweet',
                   'calm/relaxing/serene', 'compassionate/sympathetic',
                   'dreamy', 'eerie/mysterious', 'energizing/pump-up',
                   'entrancing', 'erotic/desirous', 'euphoric/ecstatic',
                   'exciting', 'goose bumps', 'indignant/defiant',
                   'joyful/cheerful', 'nauseating/revolting', 'painful',
                   'proud/strong', 'romantic/loving', 'sad/depressing',
                   'scary/fearful', 'tender/longing', 'transcendent/mystical',
                   'triumphant/heroic']
    
    df = pd.read_csv(data_path)
    
    # Modify dataset creation to include resampling
    dataset_dict = {
        "audio": [],
        "sample_rate": []
    }
    
    # Process audio files with resampling
    for file in df['sample']:
        audio_path = f"{audio_dir}/{file}"
        resampled_audio = resample_audio(audio_path)
        dataset_dict["audio"].append(resampled_audio)
        dataset_dict["sample_rate"].append(24000)
    
    # Add labels
    labels = df.iloc[:, 1:].values
    for idx, emotion in enumerate(emotion_cols):
        dataset_dict[emotion] = labels[:, idx].tolist()
    
    # Convert to HF Dataset
    dataset = Dataset.from_dict(dataset_dict).cast_column("audio", Audio())

    # Calculate split sizes (80/10/10)
    total_size = len(dataset)
    test_valid_size = 0.2  # 20% total for test and validation combined
    test_size = 0.1  # 10% of total dataset
    valid_size = 0.1  # 10% of total dataset

    # Create all splits in one go
    splits = dataset.train_test_split(
        train_size=0.8,
        test_size=0.2,
        seed=42,
        shuffle=True
    )

    # Split the test portion into validation and test
    remaining_splits = splits['test'].train_test_split(
        test_size=0.5,  # Split the remaining 20% equally
        seed=42,
        shuffle=True
    )

    # Create final dataset dictionary
    dataset_dict = {
        'train': splits['train'],
        'validation': remaining_splits['train'],
        'test': remaining_splits['test']
    }

    # Verify split sizes
    print(f"Total dataset size: {total_size}")
    print(f"Train split size: {len(dataset_dict['train'])}")
    print(f"Validation split size: {len(dataset_dict['validation'])}")
    print(f"Test split size: {len(dataset_dict['test'])}")

    # Return the dataset dictionary containing all splits
    return dataset_dict


if __name__ == "__main__":
    data_config = DataConfig()
    dataset = create_hf_dataset(data_config.ratings_file, data_config.audio_dir)
    dataset.push_to_hub("baobaoh/13-dimensions-music-emotions")
    print(dataset['train'][0])
