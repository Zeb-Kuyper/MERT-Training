from pathlib import Path
import pandas as pd
from typing import Union, List, Dict
from datasets import load_dataset
from config import DataConfig
import numpy as np

def load_data_from_csv(csv_path: str) -> List[Dict]:
    """Load data from CSV with columns: audio_path, label"""
    df = pd.read_csv(csv_path)
    return df.to_dict('records')

def load_data_from_dirs(root_dir: str) -> List[Dict]:
    """Load data from directory structure where subdirs are class names"""
    data = []
    root = Path(root_dir)
    for class_dir in root.iterdir():
        if class_dir.is_dir():
            label = class_dir.name
            for audio_file in class_dir.glob("*.mp3"):
                data.append({
                    "audio_path": str(audio_file),
                    "label": label
                })
    return data

def load_data_from_hf(dataset_name: str, split: str = "train") -> List[Dict]:
    """Load data from HuggingFace dataset"""
    dataset = load_dataset(dataset_name, split=split)
    return [
        {
            "audio_path": item["audio"]["path"],
            "label": item["label"]
        }
        for item in dataset
    ]

def prepare_dataset(data_config: DataConfig):
    """Prepare emotion classification dataset with 28 dimensional ratings"""
    df = pd.read_csv(data_config.ratings_file)
    
    # Get emotion column names from CSV (excluding 'sample' column)
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
    
    dataset = []
    for audio_path in data_config.audio_dir.glob("*.mp3"):
        file_id = audio_path.name
        if file_id in df['sample'].values:
            # Get all 28 emotion ratings for this sample
            ratings = df[df['sample'] == file_id][emotion_cols].values[0]
            dataset.append({
                'audio_path': str(audio_path),
                'label': ratings.astype(np.float32)
            })
    
    if not dataset:
        raise ValueError(f"No valid audio files found in {data_config.audio_dir}")
        
    return dataset