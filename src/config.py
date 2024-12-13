from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    target_length: int = 16000 * 10  # 10 seconds

@dataclass
class ModelConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_length: int = 16000 * 10 
    hidden_size: int = 1024
    num_classes: int = 2
    dropout: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data" / "raw"
    audio_dir: Path = project_root / "audio_samples"
    ratings_file: Path = data_dir / "MeanCategoryRatingsUSA.csv"
    model_save_path: Path = project_root / "models" / "mert_classifier.pt"
    format: str = "csv"  # Add format field
    num_classes: int = 28  # Number of emotion classes
