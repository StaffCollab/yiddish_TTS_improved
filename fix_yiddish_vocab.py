#!/usr/bin/env python3
"""
Fix Yiddish Vocabulary Issue
Creates a proper training setup that includes Hebrew characters in vocabulary
"""

import os
import json
import torch
import unicodedata
import re
from pathlib import Path


def analyze_vocabulary_issue():
    """Analyze why characters are being discarded"""
    
    print("=== Analyzing Yiddish Vocabulary Issue ===")
    
    # Load our prepared data
    if not os.path.exists("yiddish_train_data.txt"):
        print("Error: Run prepare_yiddish_data.py first!")
        return
    
    # Read our Yiddish text
    with open("yiddish_train_data.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Extract all text
    all_text = ""
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            all_text += parts[1] + " "
    
    # Get unique characters
    unique_chars = sorted(set(all_text))
    hebrew_chars = [c for c in unique_chars if ord(c) >= 0x0590 and ord(c) <= 0x05FF]
    
    print(f"Total unique characters: {len(unique_chars)}")
    print(f"Hebrew script characters: {len(hebrew_chars)}")
    print(f"Hebrew characters: {''.join(hebrew_chars)}")
    
    # Check what pre-trained models support
    print("\n=== Checking Model Vocabularies ===")
    
    try:
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.tts.configs.shared_configs import CharactersConfig
        
        # Create a custom character config for Yiddish
        char_config = CharactersConfig(
            characters="".join(hebrew_chars),
            punctuations="!?.,;: ",
            phonemes="",
            pad="_",
            eos="~",
            bos="^",
            blank="|",
            characters_class="TTS.tts.utils.text.characters.Phonemes",
        )
        
        print("✓ Created custom character config for Yiddish")
        print(f"  Characters: {''.join(hebrew_chars)}")
        
        return char_config, hebrew_chars
        
    except Exception as e:
        print(f"✗ Error with TTS tokenizer: {e}")
        return None, hebrew_chars


def create_custom_yiddish_config():
    """Create a custom TTS config that includes Hebrew characters"""
    
    print("\n=== Creating Custom Yiddish Training Config ===")
    
    # Analyze our vocabulary
    char_config, hebrew_chars = analyze_vocabulary_issue()
    
    if char_config is None:
        print("Creating basic config without TTS classes...")
        hebrew_chars_str = "".join(hebrew_chars) if hebrew_chars else "אבגדהוזחטיךכלםמןנסעףפץצקרשת"
    else:
        hebrew_chars_str = "".join(hebrew_chars)
    
    # Create comprehensive config
    yiddish_config = {
        "model": "tacotron2",
        "run_name": "yiddish_tacotron2",
        "epochs": 1000,
        "lr": 1e-3,
        "batch_size": 8,  # Smaller for CPU
        "r": 1,
        "grad_clip": 1.0,
        "seq_len_norm": False,
        
        # Audio config
        "audio": {
            "fft_size": 1024,
            "sample_rate": 22050,
            "frame_shift_ms": None,
            "frame_length_ms": None,
            "hop_length": 256,
            "win_length": 1024,
            "min_level_db": -100,
            "ref_level_db": 20,
            "power": 1.5,
            "preemphasis": 0.97,
            "n_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": 8000.0,
        },
        
        # Dataset config
        "datasets": [{
            "name": "yiddish_custom",
            "path": "./",
            "meta_file_train": "yiddish_train_data.txt",
            "meta_file_val": "yiddish_train_data.txt",  # We'll split this later
        }],
        
        # Character config - THIS IS THE KEY PART
        "characters": {
            "characters_class": "TTS.tts.utils.text.characters.Phonemes",
            "vocab_size": len(hebrew_chars_str) + 20,  # Extra space for special tokens
            "characters": hebrew_chars_str,
            "punctuations": "!?.,;:()[]{}\"'- ",
            "phonemes": "",
            "pad": "_",
            "eos": "~", 
            "bos": "^",
            "blank": "|",
            "is_unique": True,
            "is_sorted": True
        },
        
        # Text processing
        "text": {
            "text_cleaner": "basic_cleaners",
            "language": "yi",  # Yiddish language code
            "phoneme_language": "",
            "add_blank": True,
            "use_phonemes": False,
        },
        
        # Training
        "train_split": 0.9,
        "val_split": 0.1,
        "test_split": 0.0,
        "print_step": 25,
        "plot_step": 100,
        "save_step": 500,
        "save_n_checkpoints": 5,
        "save_checkpoints": True,
        
        # Output
        "output_path": "./yiddish_training_output/",
        "project_name": "YiddishTTS",
        "run_description": "Training Yiddish TTS with Hebrew script support"
    }
    
    # Save config
    config_file = "yiddish_custom_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(yiddish_config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved custom config: {config_file}")
    print(f"  Includes {len(hebrew_chars_str)} Hebrew characters")
    print(f"  Total vocab size: {len(hebrew_chars_str) + 20}")
    
    return config_file


def create_simple_training_approach():
    """Create a simple training approach that works with Hebrew characters"""
    
    print("\n=== Alternative: Simple Character-based Approach ===")
    
    training_script = '''#!/usr/bin/env python3
"""
Simple Yiddish TTS Training - Character Based
Uses character-level processing to handle Hebrew script properly
"""

import os
import torch
import json
from pathlib import Path

def train_simple_yiddish_tts():
    """Train using basic character mapping"""
    
    print("🎯 Simple Yiddish TTS Training")
    print("This approach handles Hebrew characters directly")
    
    # Load our data
    with open("yiddish_train_data.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Extract texts and build character vocabulary
    texts = []
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) >= 2:
            texts.append(parts[1])
    
    # Build character set
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    
    char_list = sorted(list(all_chars))
    char_to_idx = {char: idx for idx, char in enumerate(char_list)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(char_list)}")
    print(f"Characters: {''.join(char_list)}")
    
    # Save vocabulary
    vocab_data = {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
        "vocab_size": len(char_list),
        "characters": char_list
    }
    
    with open("yiddish_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)
    
    print("✓ Vocabulary saved: yiddish_vocabulary.json")
    print()
    print("Next: Use this vocabulary with a custom training script")
    print("      that respects Hebrew character encoding")

if __name__ == "__main__":
    train_simple_yiddish_tts()
'''
    
    with open("train_simple_yiddish.py", "w", encoding="utf-8") as f:
        f.write(training_script)
    
    print("✓ Created: train_simple_yiddish.py")


def suggest_working_alternatives():
    """Suggest alternative approaches that will work"""
    
    print("\n=== Working Alternatives ===")
    print()
    print("The pre-trained models don't support Hebrew script. Here are solutions:")
    print()
    print("1. 🎯 TRAIN FROM SCRATCH (Recommended)")
    print("   - Use tacotron2 or VITS with custom Hebrew vocabulary")
    print("   - Full control over character handling")
    print("   - Will take longer but work properly")
    print()
    print("2. 🔧 CHARACTER MAPPING")
    print("   - Map Hebrew characters to Latin equivalents") 
    print("   - Use existing models with transliteration")
    print("   - Quick solution but may affect pronunciation")
    print()
    print("3. 📚 PHONEME-BASED APPROACH")
    print("   - Convert Yiddish to phonemes first")
    print("   - Use phoneme-based TTS models")
    print("   - Best pronunciation but requires phoneme mapping")
    print()
    print("4. 🌐 CUSTOM FINE-TUNING")
    print("   - Modify pre-trained model vocabulary")
    print("   - Add Hebrew characters to existing models")
    print("   - Technical but powerful solution")


if __name__ == "__main__":
    print("🔧 Fixing Yiddish TTS Vocabulary Issues")
    print()
    
    # Analyze the problem
    analyze_vocabulary_issue()
    
    # Create solutions
    create_custom_yiddish_config()
    create_simple_training_approach()
    
    # Show alternatives
    suggest_working_alternatives()
    
    print("\n=== Summary ===")
    print("The issue is that pre-trained models don't include Hebrew characters.")
    print("Solutions created:")
    print("  - yiddish_custom_config.json (for from-scratch training)")
    print("  - train_simple_yiddish.py (basic character approach)")
    print("  - yiddish_vocabulary.json (will be created)")
    print()
    print("Recommendation: Train from scratch with custom vocabulary!") 