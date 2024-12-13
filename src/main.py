import torch
from transformers import AutoModel, AutoConfig
from model import AudioClassifier
from dataset import AudioDataset
from config import ModelConfig, DataConfig
from train import train_model, prepare_dataset

def main():
    model_config = ModelConfig()
    data_config = DataConfig()
    try:
        # Load and modify MERT config first
        mert_config = AutoConfig.from_pretrained(
            "m-a-p/MERT-v1-330M",
            trust_remote_code=True
        )
        # Add missing attributes
        setattr(mert_config, 'conv_pos_batch_norm', True)
        
        # Load MERT model with modified config
        base_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M",
            config=mert_config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        ).to(model_config.device)
        
        # Initialize classifier
        model = AudioClassifier(
            base_model=base_model,
            num_classes=data_config.num_classes  # Use num_classes from config
        ).to(model_config.device)
        
        # Prepare emotion classification datasets
        dataset = prepare_dataset(data_config)
        train_size = int(0.8 * len(dataset))
        train_dataset = AudioDataset(dataset[:train_size], model_config)
        val_dataset = AudioDataset(dataset[train_size:], model_config)
        
        # Train and save
        trained_model = train_model(model, train_dataset, val_dataset, model_config)
        torch.save(trained_model.state_dict(), data_config.model_save_path)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()