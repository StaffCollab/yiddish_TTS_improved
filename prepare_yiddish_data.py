#!/usr/bin/env python3
"""
Yiddish Data Preparation Script for TTS Training
Prepares Yiddish text (in Hebrew script) for TTS training with Coqui TTS
"""

import os
import json
import unicodedata
import re
from pathlib import Path


class YiddishTextProcessor:
    """Custom text processor for Yiddish text (using Hebrew script)"""
    
    def __init__(self):
        # Hebrew character ranges (used for Yiddish)
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


def prepare_yiddish_dataset():
    """Prepare Yiddish dataset for TTS training"""
    
    print("=== Yiddish TTS Data Preparation ===")
    print("Note: Creating a rare Yiddish TTS dataset!")
    
    # Check dataset
    metadata_path = "tts_segments/segments_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # Load metadata
    print("\n1. Loading metadata...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    processor = YiddishTextProcessor()
    
    # Prepare training data
    print("2. Processing Yiddish text...")
    training_data = []
    all_texts = []
    
    for i, item in enumerate(metadata):
        # Get normalized text
        original_text = item['text']
        normalized_text = processor.normalize_yiddish_text(original_text)
        all_texts.append(normalized_text)
        
        # Check audio file
        audio_file = item['audio_file']
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
            continue
        
        # TTS format: audio_file|text|speaker_id
        training_data.append(f"{audio_file}|{normalized_text}|speaker_0")
        
        # Show first few examples
        if i < 3:
            print(f"   Sample {i+1}:")
            print(f"      Original: {original_text[:50]}...")
            print(f"      Normalized: {normalized_text[:50]}...")
    
    # Get unique characters for tokenizer
    unique_chars = processor.get_unique_chars(all_texts)
    print(f"\n3. Character analysis:")
    print(f"   Found {len(unique_chars)} unique characters:")
    print(f"   Characters: {''.join(unique_chars)}")
    
    # Save training file
    train_file = "yiddish_train_data.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))
    
    print(f"\n4. Dataset saved:")
    print(f"   Training file: {train_file}")
    print(f"   Total samples: {len(training_data)}")
    
    # Create simple config for reference
    config_data = {
        "dataset_info": {
            "language": "Yiddish",
            "script": "Hebrew",
            "total_samples": len(training_data),
            "unique_characters": len(unique_chars),
            "characters": ''.join(unique_chars)
        },
        "training_params": {
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "batch_size": 16
        }
    }
    
    config_file = "yiddish_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    print(f"   Config file: {config_file}")
    
    # Show training commands
    print(f"\n=== Next Steps ===")
    print("Your Yiddish dataset is ready! Here's how to train:")
    print()
    print("Option 1 - Use TTS command line (recommended):")
    print("tts --model_name tts_models/en/ljspeech/tacotron2-DDC \\")
    print("    --vocoder_name vocoder_models/en/ljspeech/hifigan_v2 \\")
    print(f"    --dataset_path ./ \\")
    print(f"    --dataset_config_path {config_file} \\")
    print("    --out_path ./yiddish_tts_output")
    print()
    print("Option 2 - Train from scratch:")
    print("python -m TTS.bin.train_tts \\")
    print("    --config_path tacotron2_yiddish_config.json \\")
    print("    --restore_path '' \\")
    print("    --out_path ./yiddish_tts_output")
    
    return training_data, unique_chars


def test_sample_processing():
    """Test processing on a few samples"""
    
    print("\n=== Testing Yiddish Text Processing ===")
    
    # Test with some sample texts
    processor = YiddishTextProcessor()
    
    # Read a few samples from your data
    try:
        with open('tts_segments/text/segment_0001.txt', 'r', encoding='utf-8') as f:
            sample1 = f.read().strip()
        
        with open('tts_segments/text/segment_0010.txt', 'r', encoding='utf-8') as f:
            sample2 = f.read().strip()
        
        samples = [sample1, sample2]
        
        for i, text in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Original: {text}")
            normalized = processor.normalize_yiddish_text(text)
            print(f"  Normalized: {normalized}")
            print(f"  Length: {len(normalized)} chars")
            
    except FileNotFoundError as e:
        print(f"Could not read sample files: {e}")


if __name__ == "__main__":
    # Test text processing first
    test_sample_processing()
    
    # Prepare full dataset
    prepare_yiddish_dataset() 