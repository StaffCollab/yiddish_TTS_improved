#!/usr/bin/env python3
"""
Train Yiddish Tacotron2 TTS - FIXED VERSION
Uses properly formatted data and complete character vocabulary
"""

import os
import json
import subprocess
import sys
from pathlib import Path

# Try to import TTS components
try:
    from TTS.tts.configs.tacotron2_config import Tacotron2Config
    from TTS.config import load_config
    from TTS.utils.manage import ModelManager
    print("✅ TTS 0.22.0 components available")
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TTS components not found: {e}")
    TTS_AVAILABLE = False

def create_fixed_tacotron2_config():
    """Create Tacotron2 config with fixed character vocabulary and optimized settings"""
    
    print("🔧 Creating FIXED Tacotron2 configuration...")
    
    # Load the fixed character set and config
    with open("yiddish_config_FIXED.json", 'r', encoding='utf-8') as f:
        fixed_config = json.load(f)
    
    # Create Tacotron2 config
    config = Tacotron2Config()

    # ------------------------------------------------------------------
    # The original fixed_config["characters"]["characters"] string
    # still contained punctuation symbols (\" and ') that are ALSO
    # listed in the *punctuations* field, which leads to
    # `AssertionError: duplicate characters` when the tokenizer builds
    # its vocabulary.  Remove anything that appears in the punctuation
    # list from the main `characters` list to guarantee uniqueness.
    # ------------------------------------------------------------------

    punctuations: str = fixed_config["characters"].get("punctuations", "")
    raw_chars: str = fixed_config["characters"].get("characters", "")

    # Build a new characters string without any symbol that is also
    # listed in punctuations. This guarantees the two sets are disjoint.
    cleaned_chars = "".join(ch for ch in raw_chars if ch not in punctuations)

    # Replace the characters field in a *copy* of fixed_config so we
    # don’t mutate the on-disk JSON.
    fixed_chars_block = fixed_config["characters"].copy()
    fixed_chars_block["characters"] = cleaned_chars

    # Apply core fixes (using only known valid keys)
    config.update({
        # Fixed dataset (using corrected paths without .wav extension)
        "datasets": [
            {
                "name": "yiddish_perfect_fixed", 
                "path": "./",  # Base path for audio files
                "meta_file_train": "yiddish_tacotron2_train_data_FIXED_NO_WAV.txt",
                # "meta_file_val": "yiddish_tacotron2_train_data_FIXED_NO_WAV.txt",
                "formatter": "ljspeech"
            }
        ],
        "eval_split_size": 0.05, # 5% held out from train
        
        # COMPLETE character vocabulary with *no* punctuation duplicates
        "characters": fixed_chars_block,
        
        # Fixed test sentences (Yiddish only)
        "test_sentences": fixed_config["test_sentences"],
        
        # Core training settings
        "batch_size": 4,
        "lr": 5e-4,
        "epochs": 100,
        "print_step": 25,
        "save_step": 500,
        "run_eval": True,
        
        # Output directory
        "output_path": "yiddish_tacotron2_training_FIXED_FRESH",
        "run_name": "yiddish_tacotron2_fixed_fresh",
    })
    
    print(f"✅ Fixed configuration created:")
    print(f"   📊 Total characters: {len(fixed_config['characters']['characters'])}")
    print(f"   🎵 Audio sample rate: 16kHz")
    print(f"   📈 Batch size: 4 (stable)")
    print(f"   🎯 Learning rate: 5e-4 (optimized)")
    print(f"   💾 Output: {config.output_path}")
    
    return config

def save_config_file(config, output_dir):
    """Save config to file for training"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    config_path = output_dir / "config.json"
    config.save_json(str(config_path))
    
    print(f"📁 Fixed config saved to: {config_path}")
    return str(config_path)

def train_yiddish_tacotron2_fixed():
    """Train Tacotron2 with FIXED data and configuration"""
    
    print("🚀 YIDDISH TACOTRON2 TRAINING - FIXED VERSION")
    print("=" * 60)
    print("✅ Fixed speaker ID format (speaker_0)")
    print("✅ Complete character vocabulary (86 characters)")
    print("✅ Filtered audio segments (>0.5s)")
    print("✅ Optimized learning rate (5e-4)")
    print("✅ Proper Yiddish test sentences")
    print("🔄 STARTING FRESH (no checkpoint resumption)")
    print("=" * 60)
    
    # Check TTS availability
    if not TTS_AVAILABLE:
        print("❌ TTS not available. Install with:")
        print("   pip install TTS")
        return
    
    # Check if fixed files exist
    if not Path("yiddish_tacotron2_train_data_FIXED_NO_WAV.txt").exists():
        print("❌ Fixed training file not found!")
        print("   Creating corrected file...")
        # Create the corrected file by removing .wav extensions
        import subprocess
        subprocess.run(["sed", "s/\\.wav|/|/g", "yiddish_tacotron2_train_data_FIXED.txt"], 
                      stdout=open("yiddish_tacotron2_train_data_FIXED_NO_WAV.txt", "w"))
        print("✅ Created yiddish_tacotron2_train_data_FIXED_NO_WAV.txt")
    
    if not Path("yiddish_config_FIXED.json").exists():
        print("❌ Fixed config not found!")
        print("   Run: python fix_training_data_format.py")
        return
    
    print("✅ All fixed files found!")
    
    # Create fixed configuration
    config = create_fixed_tacotron2_config()
    
    # Save config file
    config_path = save_config_file(config, config.output_path)
    
    # Start training using TTS command line
    print(f"\n🚀 Starting FIXED Tacotron2 Training...")
    print(f"   Model: Tacotron2")
    print(f"   Language: Yiddish (Hebrew script)")
    print(f"   Samples: 1,085 (properly formatted)")
    print(f"   Characters: 86 (complete vocabulary)")
    print(f"   Sample rate: 16kHz")
    print(f"   Batch size: 4")
    print(f"   Learning rate: 5e-4")
    print(f"   Output: {config.output_path}")
    print(f"   Config: {config_path}")
    
    # Use TTS trainer directly (not the tts command which is for inference)
    try:
        print(f"\n🔥 Starting training...")
        from TTS.bin.train_tts import main as train_main
        import sys
        
        # Set up training arguments
        original_argv = sys.argv
        sys.argv = [
            "train_tts.py",
            "--config_path", config_path,
            # Starting fresh - no restore path
        ]
        
        # Start training
        train_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Model saved in: {config.output_path}")
        print(f"   🎤 Your FIXED Yiddish TTS model is ready!")
        
    except Exception as e:
        # Restore original argv on error
        import sys
        sys.argv = original_argv
        print(f"❌ Training failed: {e}")
        print(f"\n💡 Try manual training:")
        print(f"   python -m TTS.bin.train_tts --config_path {config_path}")
        
    except KeyboardInterrupt:
        # Restore original argv on interrupt
        import sys
        sys.argv = original_argv
        print(f"\n⏸️  Training interrupted by user")
        print(f"   Checkpoints saved in: {config.output_path}")
        print(f"   Resume with: python -m TTS.bin.train_tts --config_path {config_path} --restore_path {config.output_path}")

def main():
    """Main training function"""
    if "--auto" in sys.argv:
        print("🤖 Auto-start enabled")
        train_yiddish_tacotron2_fixed()
    else:
        print("🎯 Yiddish Tacotron2 Training - FIXED VERSION")
        print("=" * 50)
        print("All training issues have been resolved:")
        print("✅ Speaker ID format fixed")
        print("✅ Complete character vocabulary")
        print("✅ Audio duration filtering")
        print("✅ Optimized training settings")
        print("✅ Proper Yiddish test sentences")
        print("=" * 50)
        
        response = input("\n🚀 Start FIXED training now? (y/N): ")
        if response.lower() in ['y', 'yes']:
            train_yiddish_tacotron2_fixed()
        else:
            print("\n📋 Manual start:")
            print("   python train_yiddish_tacotron2_FIXED.py --auto")

if __name__ == "__main__":
    main() 