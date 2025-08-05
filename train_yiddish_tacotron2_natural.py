#!/usr/bin/env python3
"""
Train Yiddish Tacotron2 with Natural Segments Dataset
"""

import subprocess
import sys
from pathlib import Path

def train_natural_yiddish():
    """Train Tacotron2 with the new natural segments dataset"""
    
    # Verify that all required files exist
    config_file = Path("final_yiddish_config.json")
    train_data_file = Path("yiddish_tacotron2_natural_train_data.txt")
    segments_dir = Path("natural_mapped_segments")
    
    if not config_file.exists():
        print(f"❌ Config file {config_file} not found!")
        return False
    
    if not train_data_file.exists():
        print(f"❌ Training data file {train_data_file} not found!")
        return False
        
    if not segments_dir.exists():
        print(f"❌ Segments directory {segments_dir} not found!")
        return False
    
    print("🎯 TRAINING YIDDISH TACOTRON2 WITH NATURAL SEGMENTS")
    print("=" * 60)
    print(f"✅ Config: {config_file}")
    print(f"✅ Training data: {train_data_file}")
    print(f"✅ Segments: {segments_dir} (367 natural segments)")
    print()
    
    # Build training command
    cmd = [
        "python", "-m", "TTS.bin.train_tts",
        "--config_path", str(config_file),
        "--restore_path", "",  # Start fresh
        "--continue_path", "",  # No checkpoint
    ]
    
    print("🚀 Starting training with command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True)
        
        print("🎉 Training completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False

if __name__ == "__main__":
    success = train_natural_yiddish()
    sys.exit(0 if success else 1) 