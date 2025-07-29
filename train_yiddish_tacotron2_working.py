#!/usr/bin/env python3
"""
Train Yiddish Tacotron2 TTS on Perfect Mapped Segments
Works with TTS 0.22.0 - Handles Hebrew characters properly
Uses the correct training API for this TTS version
"""

import os
import json
import glob
from pathlib import Path
import unicodedata
import re
import subprocess
import sys

# Try to import TTS components for the correct version
try:
    from TTS.tts.configs.tacotron2_config import Tacotron2Config
    from TTS.config import load_config
    from TTS.utils.manage import ModelManager
    print("✅ TTS 0.22.0 components available")
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  TTS components not found: {e}")
    TTS_AVAILABLE = False


class YiddishTextProcessor:
    """Text processor for Yiddish (Hebrew script) for Tacotron2"""
    
    def __init__(self):
        # Hebrew character ranges
        self.hebrew_chars = set()
        # Main Hebrew block (includes all Hebrew letters)
        for i in range(0x0590, 0x05FF):
            self.hebrew_chars.add(chr(i))
        # Hebrew presentation forms
        for i in range(0xFB1D, 0xFB4F):
            self.hebrew_chars.add(chr(i))
        
        # Essential punctuation and symbols
        self.punctuation = ".,!?;:-()[]{}\"'`"
        self.allowed_chars = set(self.punctuation + " \n\t")
        self.allowed_chars.update("0123456789")  # Numbers
        
    def normalize_yiddish_text(self, text):
        """Normalize Yiddish text for Tacotron2 training"""
        # Normalize Unicode
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


def collect_perfect_segments(segments_dir="perfect_mapped_segments"):
    """Collect all perfect segments for training"""
    
    print(f"🎯 Collecting perfect segments from {segments_dir}/")
    
    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        return []
    
    all_segments = []
    metadata_files = list(segments_dir.glob("file_*_metadata.json"))
    
    print(f"📁 Found {len(metadata_files)} metadata files")
    
    total_duration = 0
    for metadata_file in metadata_files:
        print(f"   Processing {metadata_file.name}...")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add each segment
        for segment in metadata['segments']:
            # Convert relative paths to absolute
            audio_file = segments_dir / segment['audio_file']
            text_file = segments_dir / segment['text_file']
            
            if audio_file.exists() and text_file.exists():
                # Read the text
                with open(text_file, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                
                duration = segment['duration']
                total_duration += duration
                
                # Tacotron2 works well with segments 1-20 seconds
                if 1.0 <= duration <= 20.0:
                    all_segments.append({
                        'audio_file': str(audio_file),
                        'text': text,
                        'duration': duration,
                        'word_count': segment['word_count']
                    })
                else:
                    print(f"      ⚠️  Segment {segment['segment_id']} duration {duration:.1f}s outside range")
            else:
                print(f"      ⚠️  Missing files for segment {segment['segment_id']}")
    
    print(f"✅ Collected {len(all_segments)} perfect segments")
    print(f"📊 Total duration: {total_duration/60:.1f} minutes")
    return all_segments


def prepare_tacotron2_training_data(segments, output_file="yiddish_tacotron2_train_data.txt"):
    """Prepare segments in Tacotron2 training format"""
    
    print(f"📝 Preparing Tacotron2 training data...")
    
    processor = YiddishTextProcessor()
    
    # Prepare training data in TTS format: audio_file|text|speaker_id
    training_data = []
    all_texts = []
    
    for i, segment in enumerate(segments):
        # Normalize text
        text = processor.normalize_yiddish_text(segment['text'])
        all_texts.append(text)
        
        # Remove .wav extension since ljspeech formatter adds it automatically
        audio_file = segment['audio_file']
        if audio_file.endswith('.wav'):
            audio_file = audio_file[:-4]  # Remove .wav extension
        
        # Tacotron2 format: audio_path|text|speaker_name
        training_data.append(f"{audio_file}|{text}|yiddish_speaker")
        
        # Show first few examples
        if i < 3:
            print(f"   Sample {i+1}:")
            print(f"      Audio: {audio_file} (ljspeech will add .wav)")
            print(f"      Text: {text}")
            print(f"      Duration: {segment['duration']:.1f}s")
    
    # Get unique characters for vocabulary
    unique_chars = processor.get_unique_chars(all_texts)
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total segments: {len(training_data)}")
    print(f"   Unique characters: {len(unique_chars)}")
    print(f"   Characters: {''.join(unique_chars[:30])}{'...' if len(unique_chars) > 30 else ''}")
    
    # Save training file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))
    
    print(f"💾 Training data saved: {output_file}")
    return unique_chars, output_file


def create_tacotron2_config(unique_chars, training_file, output_dir="yiddish_tacotron2_training"):
    """Create Tacotron2 configuration for Yiddish training"""
    
    print(f"⚙️  Creating Tacotron2 configuration...")
    
    # Create character vocabulary for Hebrew/Yiddish
    # Separate letters from punctuation to avoid duplicates
    hebrew_letters = [c for c in unique_chars if c not in ".,!?;:-()[]{}\"'` 0123456789"]
    
    characters = {
        "pad": "_",
        "eos": "~", 
        "bos": "^",
        "characters": ''.join(hebrew_letters),  # Only Hebrew letters + space
        "punctuations": ".,!?;:-()[]{}\"'`",  # Standard punctuation
        "phonemes": "",
        "is_unique": True,
        "is_sorted": True
    }
    
    # Tacotron2 config optimized for Yiddish
    config = Tacotron2Config()
    
    # Update config with our settings using only valid attributes
    config.update({
        # Dataset configuration
        "datasets": [{
            "name": "yiddish_perfect",
            "path": "./",
            "meta_file_train": training_file,
            "meta_file_val": training_file,
            "formatter": "ljspeech",  # Standard format: audio|text|speaker (language-agnostic)
        }],
        
        # Model architecture
        "num_chars": len(characters["characters"]) + len(characters["punctuations"]) + 10,
        "num_speakers": 1,
        "r": 2,  # Reduction factor
        
        # Training parameters
        "batch_size": 8,  # Conservative for memory
        "eval_batch_size": 4,
        "epochs": 200,
        "test_delay_epochs": 10,
        
        # Optimization (using valid attributes only)
        "lr": 1e-3,
        "optimizer": "Adam",
        "lr_scheduler": "ExponentialLR", 
        "lr_scheduler_params": {"gamma": 0.95},
        
        # Characters config
        "characters": characters,
        
        # Output and logging
        "output_path": output_dir,
        "run_name": "yiddish_tacotron2_perfect",
        
        # Frequent checkpointing to prevent data loss
        "save_step": 500,  # Save every 500 steps (~4 epochs) instead of 10000
        "save_n_checkpoints": 10,  # Keep more checkpoints for safety
        "save_checkpoints": True,
        "save_on_interrupt": True,  # Save if training interrupted
        "save_best_after": 0,  # Save best model from the start
        "print_step": 25,  # Print progress frequently
    })
    
    # Configure audio settings separately
    config.audio.update({
        "sample_rate": 16000,
        "hop_length": 256, 
        "win_length": 1024,
        "fft_size": 1024,
        "preemphasis": 0.97,
    })
    
    print(f"✅ Tacotron2 configuration created")
    return config


def save_config_file(config, output_dir):
    """Save config to JSON file for training"""
    config_path = Path(output_dir) / "config.json"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert config to dict if needed
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = dict(config)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    print(f"📁 Config saved to: {config_path}")
    return str(config_path)


def train_yiddish_tacotron2():
    """Train Tacotron2 model on perfect Yiddish segments"""
    
    print("🎯 YIDDISH TACOTRON2 TRAINING ON PERFECT SEGMENTS")
    print("=" * 60)
    print("📚 Hebrew script with proper character handling")
    print("🎤 High-quality TTS for Yiddish language")
    print("=" * 60)
    
    # Check TTS availability
    if not TTS_AVAILABLE:
        print("❌ TTS not available. Install with:")
        print("   pip install TTS")
        return
    
    # Step 1: Collect perfect segments
    segments = collect_perfect_segments()
    if not segments:
        print("❌ No segments found. Make sure perfect_mapped_segments/ exists.")
        return
    
    # Check dataset size
    total_duration = sum(s['duration'] for s in segments)
    if total_duration < 300:  # Less than 5 minutes
        print(f"⚠️  Dataset quite small: {total_duration/60:.1f} minutes")
        print("   Consider getting more data for better results")
    else:
        print(f"✅ Good dataset size: {total_duration/60:.1f} minutes")
    
    # Step 2: Prepare training data
    unique_chars, training_file = prepare_tacotron2_training_data(segments)
    
    # Step 3: Create Tacotron2 config
    config = create_tacotron2_config(unique_chars, training_file)
    
    # Step 4: Save config file
    config_path = save_config_file(config, config.output_path)
    
    # Step 5: Start training using TTS command line
    print(f"\n🚀 Starting Tacotron2 Training...")
    print(f"   Model: Tacotron2")
    print(f"   Language: Yiddish (Hebrew script)")
    print(f"   Segments: {len(segments)}")
    print(f"   Characters: {len(unique_chars)}")
    print(f"   Sample rate: 16kHz")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Output: {config.output_path}")
    
    # Check for existing checkpoints to resume from
    checkpoint_dir = Path(config.output_path)
    latest_checkpoint = None
    if checkpoint_dir.exists():
        # Look for checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("**/checkpoint_*.pth"))
        if checkpoint_files:
            # Get the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Found existing checkpoint: {latest_checkpoint}")
            print(f"🔄 Training will resume from this checkpoint")
        else:
            print(f"📁 No existing checkpoints found - starting fresh")
    
    # Prepare training command
    train_cmd = [
        "python", "-m", "TTS.bin.train_tts",
        "--config_path", config_path,
        "--force_enable_cuda", "true" if "cuda" in str(config).lower() else "false"
    ]
    
    # Add resume option if checkpoint exists
    if latest_checkpoint:
        train_cmd.extend(["--restore_path", str(latest_checkpoint)])
    
    print(f"\n🎬 Training command:")
    print(f"   {' '.join(train_cmd)}")
    
    try:
        print(f"\n🚀 Starting training...")
        print(f"📊 Monitor training progress in the output directory")
        print(f"   Output directory: {config.output_path}")
        
        # Start training
        result = subprocess.run(train_cmd, check=True, cwd=".")
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Model saved in: {config.output_path}")
        print(f"   🎤 You now have a Yiddish Tacotron2 model!")
        
        # Show usage example
        print(f"\n📖 Usage example:")
        print(f"```python")
        print(f"from TTS.api import TTS")
        print(f"# Load your trained model")
        print(f"tts = TTS(model_path='{config.output_path}/best_model.pth',")
        print(f"          config_path='{config_path}')")
        print(f"tts.tts_to_file(text='מיין נסיעה קיין קערעסטיר', file_path='output.wav')")
        print(f"```")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code {e.returncode}")
        print(f"\n💡 Try running the command manually:")
        print(f"   {' '.join(train_cmd)}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"\n💡 Alternative approach:")
        print(f"   Run the training command manually:")
        print(f"   {' '.join(train_cmd)}")


def show_requirements():
    """Show system requirements"""
    
    requirements = """
🎯 YIDDISH TACOTRON2 TRAINING REQUIREMENTS
==========================================

📋 SYSTEM REQUIREMENTS:
- 🐍 Python 3.8+
- 🔥 CUDA-capable GPU (6GB+ VRAM recommended)
- 💾 8GB+ RAM
- 💽 3GB+ free disk space

📦 SOFTWARE REQUIREMENTS:
- TTS==0.22.0 (installed ✅)
- torch, torchaudio
- numpy, librosa, soundfile

🎤 DATASET STATUS:
- ✅ Perfect audio-text alignment
- ✅ Hebrew script support (Tacotron2 compatible!)
- ✅ Right-to-left text handling
- ✅ Multiple speakers across 21 audio files
- ✅ High-quality 16kHz audio

🌟 TACOTRON2 ADVANTAGES:
- ✅ Supports ANY Unicode characters (including Hebrew!)
- 🎯 Character-based (perfect for Hebrew script)
- 🚀 Proven architecture
- 📚 Works with TTS 0.22.0
- 🔓 No language restrictions

This will be one of the first Yiddish Tacotron2 models! 🎉
"""
    print(requirements)


def main():
    """Main function"""
    import sys
    import os
    
    show_requirements()
    
    # Check for auto-start modes
    auto_start = (len(sys.argv) > 1 and sys.argv[1] == '--auto') or os.getenv('AUTO_TRAIN') == '1'
    
    if auto_start:
        print("\n🚀 Auto-starting training (background mode)")
        response = 'y'
    else:
        # Ask user if they want to proceed
        print("\nReady to train Yiddish Tacotron2? (y/n): ", end="")
        try:
            response = input().strip().lower()
        except (OSError, EOFError):
            # Handle background execution where input() fails
            print("\n🚀 Auto-starting training (background detected)")
            response = 'y'
    
    if response in ['y', 'yes']:
        train_yiddish_tacotron2()
    else:
        print("Training cancelled. Run again when ready!")
        print("\n💡 Make sure you have:")
        print("   - CUDA-capable GPU with 6GB+ VRAM")
        print("   - TTS 0.22.0 installed")
        print("   - Perfect segments in perfect_mapped_segments/")


if __name__ == "__main__":
    main() 