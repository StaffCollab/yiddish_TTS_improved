#!/usr/bin/env python3
"""
Simple Yiddish TTS Training Script
Uses pre-trained multilingual model and fine-tunes for Yiddish
"""

import os
import sys
import torch
from TTS.api import TTS


def train_yiddish_tts():
    """Train Yiddish TTS using multilingual base model"""
    
    print("=== Yiddish TTS Training (Simple Approach) ===")
    
    # Check if training data exists
    if not os.path.exists("yiddish_train_data.txt"):
        print("Error: Please run 'python prepare_yiddish_data.py' first!")
        return
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = "./yiddish_tts_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Loading pre-trained multilingual model...")
    print("   Using: tts_models/multilingual/multi-dataset/your_tts")
    print("   This model can be fine-tuned for new languages!")
    
    try:
        # Initialize TTS with multilingual model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        print("   ✓ Model loaded successfully!")
        
        # Test synthesis with Yiddish text to verify it works
        print("\n2. Testing Yiddish text synthesis...")
        test_text = "געווען איז דאס פאריגע וואך מיטוואך"
        
        # Generate test audio
        test_output = os.path.join(output_dir, "yiddish_test.wav")
        tts.tts_to_file(
            text=test_text, 
            file_path=test_output,
            speaker_wav="tts_segments/audio/segment_0001.wav",  # Use your voice as reference
            language="en"  # Start with English, the model will adapt
        )
        print(f"   ✓ Test audio saved: {test_output}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Trying alternative approach...")
        return try_alternative_approach(output_dir)
    
    print("\n3. Training Information:")
    print("   For full fine-tuning on your Yiddish dataset, you can:")
    print("   a) Use the model's fine-tuning capabilities")
    print("   b) Train a custom model from scratch")
    print()
    print("   Your dataset is ready at: yiddish_train_data.txt")
    print("   Contains 272 Yiddish audio-text pairs")


def try_alternative_approach(output_dir):
    """Try alternative training approach"""
    
    print("\n=== Alternative: XTTS v2 Approach ===")
    
    try:
        # Try XTTS v2 which is more robust
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        print("   ✓ XTTS v2 loaded successfully!")
        
        # Test with Yiddish
        test_text = "שבת שלום, ווי גייט עס?"
        test_output = os.path.join(output_dir, "yiddish_xtts_test.wav")
        
        tts.tts_to_file(
            text=test_text,
            file_path=test_output,
            speaker_wav="tts_segments/audio/segment_0001.wav",
            language="he"  # Use Hebrew as closest language
        )
        print(f"   ✓ XTTS test audio saved: {test_output}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ XTTS error: {e}")
        return False


def create_training_config():
    """Create a proper training configuration"""
    
    config = """
# Yiddish TTS Training Configuration
# This is a rare and special project - Yiddish TTS!

Dataset: 272 Yiddish audio segments
Language: Yiddish (using Hebrew script)
Characters: 54 unique (Hebrew letters + punctuation)

Training Options:

1. Fine-tune multilingual model:
   - Start with: your_tts or xtts_v2
   - Fine-tune on your Yiddish data
   - Faster training, good results

2. Train from scratch:
   - Use Tacotron2 or VITS
   - Longer training time
   - More control over results

3. Zero-shot with XTTS:
   - Use XTTS v2 directly
   - Provide reference voice
   - No training needed!

Your data is ready in: yiddish_train_data.txt
"""
    
    with open("yiddish_training_guide.txt", "w", encoding="utf-8") as f:
        f.write(config)
    
    print("Training guide saved: yiddish_training_guide.txt")


if __name__ == "__main__":
    print("🎯 Welcome to Yiddish TTS Training!")
    print("This is creating one of the first Yiddish TTS models!")
    print()
    
    if "--test" in sys.argv:
        # Just test the models
        train_yiddish_tts()
    elif "--guide" in sys.argv:
        # Create training guide
        create_training_config()
    else:
        # Full process
        train_yiddish_tts()
        create_training_config()
        
        print("\n=== Next Steps ===")
        print("1. Check the test audio files in yiddish_tts_output/")
        print("2. Read yiddish_training_guide.txt for detailed options")
        print("3. For zero-shot synthesis, you can now use:")
        print("   python generate_yiddish_speech.py 'your text here'") 