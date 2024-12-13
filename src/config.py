from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10
    model_name: str = "m-a-p/MERT-v1-330M"