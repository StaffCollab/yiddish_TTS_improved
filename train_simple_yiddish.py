#!/usr/bin/env python3
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
