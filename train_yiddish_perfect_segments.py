#!/usr/bin/env python3
"""
Train Yiddish TTS on Perfect Mapped Segments
Uses Coqui TTS with Tacotron2 and the perfectly aligned dataset
Handles Hebrew characters and RTL text properly
"""

import os
import json
import glob
from pathlib import Path
import unicodedata
import re

# Try to import TTS components
try:
    from TTS.tts.configs.tacotron2_config import Tacotron2Config
    from TTS.trainer import Trainer, TrainerArgs
    print("✅ Coqui TTS available")
    TTS_AVAILABLE = True
except ImportError:
    print("⚠️  Coqui TTS not found. Install with: pip install TTS")
    TTS_AVAILABLE = False


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


def collect_perfect_segments(segments_dir="perfect_mapped_segments"):
    """Collect all perfect segments into a training dataset"""
    
    print(f"🎯 Collecting perfect segments from {segments_dir}/")
    
    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        print(f"❌ Segments directory not found: {segments_dir}")
        return []
    
    all_segments = []
    metadata_files = list(segments_dir.glob("file_*_metadata.json"))
    
    print(f"📁 Found {len(metadata_files)} metadata files")
    
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
                
                all_segments.append({
                    'audio_file': str(audio_file),
                    'text': text,
                    'duration': segment['duration'],
                    'word_count': segment['word_count']
                })
            else:
                print(f"      ⚠️  Missing files for segment {segment['segment_id']}")
    
    print(f"✅ Collected {len(all_segments)} perfect segments")
    return all_segments


def prepare_tts_training_data(segments, output_file="yiddish_perfect_train_data.txt"):
    """Prepare segments in TTS training format"""
    
    print(f"📝 Preparing TTS training data...")
    
    processor = YiddishTextProcessor()
    
    # Prepare training data in TTS format: audio_file|text|speaker_id
    training_data = []
    all_texts = []
    
    for i, segment in enumerate(segments):
        # Normalize text
        text = processor.normalize_yiddish_text(segment['text'])
        all_texts.append(text)
        
        # TTS format
        audio_file = segment['audio_file']
        training_data.append(f"{audio_file}|{text}|speaker_0")
        
        # Show first few examples
        if i < 3:
            print(f"   Sample {i+1}:")
            print(f"      Audio: {Path(audio_file).name}")
            print(f"      Text: {text[:50]}...")
            print(f"      Duration: {segment['duration']:.1f}s")
    
    # Get unique characters for tokenizer
    unique_chars = processor.get_unique_chars(all_texts)
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total segments: {len(training_data)}")
    print(f"   Unique characters: {len(unique_chars)}")
    print(f"   Characters: {''.join(unique_chars)}")
    
    # Save training file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))
    
    print(f"💾 Training data saved: {output_file}")
    return unique_chars, output_file


def create_yiddish_tts_config(unique_chars, training_file, output_dir="yiddish_tts_training"):
    """Create TTS configuration for Yiddish training"""
    
    print(f"⚙️  Creating TTS configuration...")
    
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
        dataset="yiddish_perfect",
        meta_file_train=training_file,
        meta_file_val=training_file,  # Using same file, will split automatically
        path_to_dataset="./",
        
        # Model architecture
        num_chars=len(characters["characters"]) + len(characters["punctuations"]) + 10,
        num_speakers=1,
        r=2,  # Reduction factor
        
        # Training parameters
        batch_size=8,  # Conservative for memory
        eval_batch_size=4,
        num_loader_workers=2,
        num_eval_loader_workers=1,
        run_eval=True,
        test_delay_epochs=10,
        epochs=200,
        text_cleaner="basic_cleaners",
        use_phonemes=False,
        
        # Audio config
        sample_rate=16000,  # Match our perfect segments
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
        output_path=output_dir,
        run_name="yiddish_perfect_tacotron2",
        project_name="YiddishTTS_Perfect",
        run_description="Yiddish TTS training on perfectly aligned segments",
        print_step=25,
        plot_step=100,
        save_step=500,
        save_n_checkpoints=5,
        save_checkpoints=True,
        
        # Data split
        train_split=0.85,
        val_split=0.15,
    )
    
    print(f"✅ Configuration created for {output_dir}")
    return config


def train_yiddish_tts():
    """Train Yiddish TTS on perfect segments"""
    
    print("🎯 YIDDISH TTS TRAINING ON PERFECT SEGMENTS")
    print("=" * 60)
    
    # Check TTS availability
    if not TTS_AVAILABLE:
        print("❌ Coqui TTS not available. Install with:")
        print("   pip install TTS")
        return
    
    # Step 1: Collect perfect segments
    segments = collect_perfect_segments()
    if not segments:
        print("❌ No segments found. Make sure perfect_mapped_segments/ exists.")
        return
    
    # Step 2: Prepare training data
    unique_chars, training_file = prepare_tts_training_data(segments)
    
    # Step 3: Create TTS config
    config = create_yiddish_tts_config(unique_chars, training_file)
    
    # Step 4: Start training
    print(f"\n🚀 Starting TTS Training...")
    print(f"   Model: Tacotron2")
    print(f"   Language: Yiddish (Hebrew script)")
    print(f"   Segments: {len(segments)}")
    print(f"   Characters: {len(unique_chars)}")
    
    try:
        # Create trainer
        trainer_args = TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=True,
            verbose_file=True
        )
        
        trainer = Trainer(
            args=trainer_args,
            config=config,
            output_path=config.output_path,
            model=None,  # Will be created automatically
        )
        
        # Start training
        trainer.fit()
        
        print(f"\n🎉 Training completed!")
        print(f"   Model saved in: {config.output_path}")
        print(f"   🎤 You now have a Yiddish TTS model!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print(f"\n💡 Alternative approach:")
        print(f"   You can also train manually using:")
        print(f"   tts --model_name tacotron2 --train_data {training_file}")


def show_usage_guide():
    """Show how to use the trained model"""
    
    guide = """
🎯 YIDDISH TTS TRAINING GUIDE
============================

Your perfect dataset is ready! Here's what this script does:

1. ✅ Collects all 1,085 perfect segments
2. ✅ Handles Hebrew character tokenization
3. ✅ Creates proper TTS training format
4. ✅ Configures Tacotron2 for Yiddish
5. ✅ Starts training process

After training completes, you can synthesize Yiddish speech:

```python
from TTS.api import TTS
tts = TTS(model_path="./yiddish_tts_training/best_model.pth")
tts.tts_to_file(text="מיין נסיעה קיין קערעסטיר", file_path="output.wav")
```

This will be one of the first working Yiddish TTS models! 🎉
"""
    print(guide)


def main():
    """Main function"""
    show_usage_guide()
    
    # Ask user if they want to proceed
    print("\nReady to train Yiddish TTS? (y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        train_yiddish_tts()
    else:
        print("Training cancelled. Run again when ready!")


if __name__ == "__main__":
    main() 