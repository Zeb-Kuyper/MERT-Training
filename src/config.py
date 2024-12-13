from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class ModelConfig:
    model_name: str = "m-a-p/MERT-v1-330M"
    hidden_size: int = 1024
    num_classes: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data" / "raw"
    audio_dir: Path = project_root / "audio_samples"
    ratings_file: Path = data_dir / "MeanCategoryRatingsUSA.csv"
    model_save_path: Path = project_root / "models" / "MERT-v1-330M_classifier.pt"
    max_length: int = 16000 * 10  # 10 seconds of audio

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip: float = 1.0