#!/usr/bin/env python3
"""
Yiddish TTS Training Script using Coqui TTS
Handles Yiddish text (in Hebrew script) tokenization and trains on your segmented dataset
Note: This creates a rare Yiddish TTS model - most TTS systems don't support Yiddish!
"""

import os
import json
import pandas as pd
import torch
import torchaudio
from pathlib import Path
import unicodedata
import re

from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import TTSDataset
from TTS.tts.utils.speakers import SpeakerManager
from TTS.trainer import Trainer, TrainerArgs


class YiddishTextProcessor:
    """Custom text processor for Yiddish text (using Hebrew script)"""
    
    def __init__(self):
        # Hebrew character ranges
        self.hebrew_chars = set()
        # Hebrew letters (including final forms)
        for i in range(0x0590, 0x05FF):  # Hebrew block
            self.hebrew_chars.add(chr(i))
        
        # Common punctuation and symbols
        self.allowed_chars = set(".,!?;:-()[]{}\"' \n")
        self.allowed_chars.update("0123456789")  # Numbers
        
    def normalize_yiddish_text(self, text):
        """Normalize Yiddish text for TTS training"""
        # Remove or normalize diacritics
        text = unicodedata.normalize('NFD', text)
        
        # Keep only Hebrew script characters, punctuation, and spaces
        cleaned_chars = []
        for char in text:
            if char in self.hebrew_chars or char in self.allowed_chars:
                cleaned_chars.append(char)
            elif char.isspace():
                cleaned_chars.append(' ')
        
        text = ''.join(cleaned_chars)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle special Hebrew punctuation
        text = text.replace('״', '"')  # Hebrew geresh
        text = text.replace('׳', "'")  # Hebrew gershayim
        
        return text
    
    def get_unique_chars(self, texts):
        """Get unique characters from all texts"""
        unique_chars = set()
        for text in texts:
            normalized = self.normalize_yiddish_text(text)
            unique_chars.update(normalized)
        return sorted(list(unique_chars))


def prepare_dataset_file(metadata_path, output_path):
    """Convert metadata to TTS training format"""
    
    print("Loading metadata...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    processor = YiddishTextProcessor()
    
    # Prepare training data
    training_data = []
    all_texts = []
    
    for item in metadata:
        # Get normalized text
        text = processor.normalize_yiddish_text(item['text'])
        all_texts.append(text)
        
        # TTS format: audio_file|text|speaker_id
        audio_file = item['audio_file']
        
        # Use relative path from project root
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
            continue
            
        training_data.append(f"{audio_file}|{text}|speaker_0")
    
    # Get unique characters for tokenizer
    unique_chars = processor.get_unique_chars(all_texts)
    print(f"Found {len(unique_chars)} unique characters:")
    print(''.join(unique_chars))
    
    # Save training file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))
    
    print(f"Saved {len(training_data)} training samples to {output_path}")
    return unique_chars


def create_tts_config(unique_chars, output_path="yiddish_tts_config.json"):
    """Create TTS configuration for Yiddish training"""
    
    # Create character vocabulary
    characters = {
        "pad": "_",
        "eos": "~",
        "bos": "^",
        "characters": ''.join(unique_chars),
        "punctuations": "!?.,;:-()[]{}\"' ",
        "phonemes": "",
        "is_unique": True,
        "is_sorted": True
    }
    
    # Tacotron2 config optimized for Yiddish
    config = Tacotron2Config(
        # Data paths
        dataset="custom_yiddish",
        meta_file_train="train_data.txt",
        meta_file_val="train_data.txt",  # Using same file for now, will split later
        path_to_dataset="./",
        
        # Model architecture  
        num_chars=len(characters["characters"]) + len(characters["punctuations"]) + 10,
        num_speakers=1,
        r=5,  # Reduction factor
        
        # Training parameters
        batch_size=16,  # Smaller batch for CPU training
        eval_batch_size=8,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=5,
        epochs=100,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        phoneme_language="",
        
        # Audio config
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        preemphasis=0.97,
        power=1.5,
        griffinlim_iters=60,
        
        # Optimization
        lr=1e-3,
        lr_decay=True,
        step_size_up=2000,
        step_size_down=2000,
        gamma=0.5,
        weight_decay=1e-6,
        grad_clip=5.0,
        
        # Characters config
        characters=characters,
        
        # Output and logging
        output_path="./tts_training_output/",
        run_name="yiddish_tacotron2",
        project_name="YiddishTTS",
        run_description="Yiddish TTS training with Tacotron2",
        print_step=50,
        save_step=1000,
        plot_step=100,
        log_model_step=1000,
        save_n_checkpoints=5,
        save_checkpoints=True,
        target_loss="loss_1",
        dashboard_logger="tensorboard",
        
        # Mixed precision (only if CUDA available)
        mixed_precision=torch.cuda.is_available(),
    )
    
    # Save config
    config.save_json(output_path)
    print(f"Configuration saved to {output_path}")
    return config


def main():
    """Main training function"""
    
    print("=== Yiddish TTS Training Setup ===")
    
    # Check dataset
    metadata_path = "tts_segments/segments_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # Prepare training data
    print("\n1. Preparing dataset...")
    unique_chars = prepare_dataset_file(metadata_path, "train_data.txt")
    
    # Create configuration
    print("\n2. Creating TTS configuration...")
    config = create_tts_config(unique_chars)
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    print("\n3. Configuration Summary:")
    print(f"   - Dataset: {len(open('train_data.txt').readlines())} samples")
    print(f"   - Characters: {config.num_chars}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Sample rate: {config.sample_rate}")
    print(f"   - Output: {config.output_path}")
    
    print("\n=== Ready for Training ===")
    print("To start training, run:")
    print("python train_hebrew_tts.py --train")
    
    # If --train argument is provided, start training
    import sys
    if "--train" in sys.argv:
        print("\n4. Starting training...")
        start_training(config)


def start_training(config):
    """Start the actual TTS training process"""
    
    # Initialize trainer
    trainer_args = TrainerArgs(
        restore_path=None,
        skip_train_epoch=False,
        start_with_eval=True,
        seed=1234,
    )
    
    trainer = Trainer(
        trainer_args,
        config,
        output_path=config.output_path,
        gpu=0 if torch.cuda.is_available() else None,
        rank=0,
    )
    
    print("Starting training...")
    trainer.fit()


if __name__ == "__main__":
    main() 