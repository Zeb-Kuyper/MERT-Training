import torch
import torchaudio.transforms as T
from transformers import AutoModel, AutoConfig, Wav2Vec2FeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from model import AudioClassifier
from datasets import load_dataset, load_metric
from config import ModelConfig, DataConfig


def preprocess_function(samples,processor):
    audio_arrays = [x["array"] for x in samples["audio"]]
    inputs = processor(audio_arrays, samplin_rate=processor.sampling_rate, return_tensors="pt")
    return inputs

def main():
    model_config = ModelConfig()
    data_config = DataConfig()
    
    
    try:
        dataset = load_dataset("baobaoh/13-dimension-music-emotions", split="train")
        print(dataset['train'][0])
        sampling_rate = dataset.features["audio"].sampling_rate
        labels = dataset["train"].feature["label"].names
        # Load and modify MERT config first
        mert_config = AutoConfig.from_pretrained(
            "m-a-p/MERT-v1-330M",
            trust_remote_code=True
        )
        # Add missing attributes
        setattr(mert_config, 'conv_pos_batch_norm', True)
        num_labels = len(id2label)
        # Load MERT model with modified config
        base_model = AutoModelForAudioClassification.from_pretrained(
            "m-a-p/MERT-v1-330M",
            config=mert_config,
            trust_remote_code=True,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        ).to(model_config.device)
        
        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        resample_rate = processor.sampling_rate

        encoded_dataset = dataset.map(preprocess_function(dataset['train'],processor),remove_columns=["audio","file"], batched=True)
        
        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)
     
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()