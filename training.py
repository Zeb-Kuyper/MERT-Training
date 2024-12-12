import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Dict
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

def load_audio(file_path: Union[str, Path], target_sr: int = 24000) -> torch.Tensor:
    """Load and preprocess audio file"""
    try:
        waveform, sr = torchaudio.load(file_path)
    except:
        waveform, sr = librosa.load(file_path, sr=None, mono=False)
        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
    
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    
    if waveform.shape[0] > 1:  # Convert stereo to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform

class AudioMixupAugmentation:
    """In-batch audio mixup augmentation"""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, audio_batch):
        if np.random.random() > self.p:
            return audio_batch
        
        batch_size = audio_batch.size(0)
        permutation = torch.randperm(batch_size)
        mix_ratios = torch.FloatTensor(batch_size).uniform_(0.1, 0.4)
        
        mixed_batch = audio_batch.clone()
        for i in range(batch_size):
            mix_idx = permutation[i]
            if mix_idx != i:
                mixed_batch[i] += mix_ratios[i] * audio_batch[mix_idx]
        
        mixed_batch = mixed_batch / mixed_batch.abs().max(dim=-1, keepdim=True)[0]
        return mixed_batch

class MusicEmotionDataset(Dataset):
    """Dataset for emotion classification with 5-second segment handling"""
    def __init__(
        self, 
        audio_paths: List[str], 
        emotion_labels: List[np.ndarray], 
        feature_extractor,
        segment_length: int = 120000,  # 5 seconds at 24kHz
        inference_mode: bool = False,
        num_inference_segments: int = 6
    ):
        self.audio_paths = audio_paths
        self.emotion_labels = emotion_labels
        self.feature_extractor = feature_extractor
        self.segment_length = segment_length
        self.inference_mode = inference_mode
        self.num_inference_segments = num_inference_segments
        self.mixup = AudioMixupAugmentation(p=0.5) if not inference_mode else None
        
    def __len__(self):
        return len(self.audio_paths)
    
    def get_audio_segments(self, audio_path):
        """Process audio into 5-second segments"""
        waveform = load_audio(audio_path)
        total_length = waveform.shape[1]
        
        if self.inference_mode:
            # For inference, take evenly spaced segments
            if total_length <= self.segment_length:
                return waveform.repeat(1, max(1, self.segment_length // total_length))
            
            segments = []
            step = (total_length - self.segment_length) // (self.num_inference_segments - 1)
            for i in range(self.num_inference_segments):
                start = min(i * step, total_length - self.segment_length)
                segments.append(waveform[:, start:start + self.segment_length])
            return torch.cat(segments, dim=0)
        else:
            # For training, take random segment
            if total_length <= self.segment_length:
                # Pad if too short
                return waveform.repeat(1, max(1, self.segment_length // total_length))
            
            start = torch.randint(0, total_length - self.segment_length, (1,))
            return waveform[:, start:start + self.segment_length]
    
    def __getitem__(self, idx):
        if self.inference_mode:
            audio_segments = self.get_audio_segments(self.audio_paths[idx])
            inputs = self.feature_extractor(
                [segment.numpy() for segment in audio_segments],
                sampling_rate=24000,
                return_tensors="pt"
            )
            
            return {
                'input_values': inputs.input_values,
                'label': self.emotion_labels[idx]
            }
        else:
            audio = self.get_audio_segments(self.audio_paths[idx])
            inputs = self.feature_extractor(
                audio.numpy(),
                sampling_rate=24000,
                return_tensors="pt"
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'label': torch.tensor(self.emotion_labels[idx], dtype=torch.float32)
            }

class EmotionClassifier(nn.Module):
    """Wrapper for MERT model with multi-segment inference handling"""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, input_values):
        if self.training:
            return self.base_model(input_values)
        else:
            if len(input_values.shape) == 3:
                batch_size, num_segments, length = input_values.shape
                input_values = input_values.view(-1, length)
                outputs = self.base_model(input_values)
                logits = outputs.logits.view(batch_size, num_segments, -1)
                outputs.logits = logits.mean(1)
                return outputs
            return self.base_model(input_values)

def train_emotion_classifier(
    train_dataset: MusicEmotionDataset,
    val_dataset: MusicEmotionDataset,
    num_labels: int,
    model_name: str = "m-a-p/MERT-v1-330M",
    batch_size: int = 16,            # Reduced from 32
    gradient_accumulation_steps: int = 2,  # New parameter
    max_audio_length_s: float = 10.0,  # Max audio length in seconds
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    gradient_clip_val: float = 1.0,
    device: str = 'cuda',
    output_dir: str = 'emotion_model',
    use_8bit: bool = False,          # Enable 8-bit training
    mixed_precision: bool = True     # Enable mixed precision
):
    """Train emotion classifier with memory optimizations for RTX 4070"""
    
    # Model initialization with memory optimizations
    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="regression",
        use_cache=False,  # Disable KV cache
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Optional 8-bit quantization
    if use_8bit:
        model = model.to(8)
    
    model = model.to(device)
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if mixed_precision else None
    
    # DataLoader with dynamic batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

def predict_emotion(model, audio_path, feature_extractor, device='cuda'):
    """Inference function for single audio prediction"""
    dataset = MusicEmotionDataset(
        [audio_path],
        [np.zeros(1)],  # Dummy label
        feature_extractor,
        inference_mode=True
    )
    
    model.eval()
    with torch.no_grad():
        batch = dataset[0]
        inputs = batch['input_values'].to(device)
        outputs = model(inputs)
        prediction = outputs.logits.mean(0)
        
    return prediction

def normalize_ratings(ratings, min_val=1, max_val=9):
    """Normalize Likert scale ratings to [0,1]"""
    return (ratings - min_val) / (max_val - min_val)

def organize_audio_files():
    # Source directories
    macosx_dir = Path("_MACOSX/Verified_Normed")
    verified_dir = Path("Verified_Normed")
    
    # Target directory
    target_dir = Path("audio_samples")
    target_dir.mkdir(exist_ok=True)
    
    # Move files from Verified_Normed
    for audio_file in verified_dir.glob("*.mp3"):
        shutil.move(str(audio_file), str(target_dir / audio_file.name))
    
    # Cleanup
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)
    if verified_dir.exists():
        shutil.rmtree(verified_dir)

def load_dataset_info(data_root: str, csv_path: str) -> tuple:
    """Load audio paths and emotion ratings from CSV"""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get audio paths and labels
    audio_paths = []
    emotion_labels = []
    
    for index, row in df.iterrows():
        audio_file = row['sample']
        file_path = os.path.join(data_root, audio_file)
        if os.path.exists(file_path):
            audio_paths.append(file_path)
            # Convert emotion ratings to tensor (excluding 'sample' column)
            labels = row[1:].values.astype(np.float32)
            emotion_labels.append(labels)
    
    return audio_paths, emotion_labels, len(df.columns) - 1  # -1 for 'sample' column

def main():
    # Set paths
    data_root = "./audio_samples/*.mp3"
    csv_path = "./data/raw/MeanCategoryRatingsUSA.csv"
    output_dir = "./checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    audio_paths, emotion_labels, num_emotions = load_dataset_info(data_root, csv_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M")

    # Create datasets
    train_dataset = MusicEmotionDataset(
        audio_paths=audio_paths[:800],  # 80% for training
        emotion_labels=emotion_labels[:800],
        feature_extractor=feature_extractor
    )

    val_dataset = MusicEmotionDataset(
        audio_paths=audio_paths[800:],  # 20% for validation
        emotion_labels=emotion_labels[800:],
        feature_extractor=feature_extractor
    )

    # Train
    model = train_emotion_classifier(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_labels=num_emotions,
        batch_size=16,
        gradient_accumulation_steps=2,
        num_epochs=10,
        output_dir=output_dir,
        mixed_precision=True
    )

if __name__ == "__main__":
    main()