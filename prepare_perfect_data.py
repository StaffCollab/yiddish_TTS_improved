#!/usr/bin/env python3
"""
Prepare Perfect Mapped Segments for TTS Training
Converts our perfect segments into the format expected by existing TTS scripts
"""

import os
import json
import shutil
from pathlib import Path
import unicodedata
import re


class YiddishTextProcessor:
    """Text processor for Yiddish text (Hebrew script)"""
    
    def __init__(self):
        # Hebrew character ranges
        self.hebrew_chars = set()
        for i in range(0x0590, 0x05FF):  # Hebrew block
            self.hebrew_chars.add(chr(i))
        
        # Allowed punctuation and symbols
        self.allowed_chars = set(".,!?;:-()[]{}\"' \n")
        self.allowed_chars.update("0123456789")
        
    def normalize_yiddish_text(self, text):
        """Normalize Yiddish text for TTS training"""
        text = unicodedata.normalize('NFD', text)
        
        # Keep only Hebrew script characters, punctuation, and spaces
        cleaned_chars = []
        for char in text:
            if char in self.hebrew_chars or char in self.allowed_chars:
                cleaned_chars.append(char)
            elif char.isspace():
                cleaned_chars.append(' ')
        
        text = ''.join(cleaned_chars)
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
    """Collect all perfect segments"""
    
    print(f"🎯 Collecting perfect segments from {segments_dir}/")
    
    segments_dir = Path(segments_dir)
    if not segments_dir.exists():
        print(f"❌ Directory not found: {segments_dir}")
        return []
    
    all_segments = []
    metadata_files = list(segments_dir.glob("file_*_metadata.json"))
    
    print(f"📁 Found {len(metadata_files)} metadata files")
    
    for metadata_file in metadata_files:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        for segment in metadata['segments']:
            # Get absolute paths
            audio_file = segments_dir / segment['audio_file']
            text_file = segments_dir / segment['text_file']
            
            if audio_file.exists() and text_file.exists():
                # Read text
                with open(text_file, 'r', encoding='utf-8') as tf:
                    text = tf.read().strip()
                
                all_segments.append({
                    'audio_file': str(audio_file),
                    'text': text,
                    'duration': segment['duration'],
                    'word_count': segment['word_count'],
                    'segment_id': segment['segment_id']
                })
    
    print(f"✅ Collected {len(all_segments)} perfect segments")
    return all_segments


def create_tts_segments_directory(segments, output_dir="tts_segments"):
    """Create the directory structure expected by existing TTS scripts"""
    
    print(f"📁 Creating TTS segments in {output_dir}/")
    
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # Create directory structure
    audio_dir = output_path / "audio"
    text_dir = output_path / "text"
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    processor = YiddishTextProcessor()
    
    # Copy and process segments
    processed_segments = []
    all_texts = []
    
    for i, segment in enumerate(segments, 1):
        segment_id = f"{i:04d}"
        
        # Copy audio file
        audio_src = Path(segment['audio_file'])
        audio_dst = audio_dir / f"segment_{segment_id}.wav"
        shutil.copy2(audio_src, audio_dst)
        
        # Process and save text
        text = processor.normalize_yiddish_text(segment['text'])
        all_texts.append(text)
        
        text_dst = text_dir / f"segment_{segment_id}.txt"
        with open(text_dst, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Create metadata entry
        processed_segments.append({
            'segment_id': segment_id,
            'audio_file': f"audio/segment_{segment_id}.wav",
            'text_file': f"text/segment_{segment_id}.txt",
            'text': text,
            'duration': segment['duration'],
            'word_count': segment['word_count']
        })
    
    # Create segments metadata
    segments_metadata = {
        'total_segments': len(processed_segments),
        'total_words': sum(seg['word_count'] for seg in segments),
        'method': 'perfect_whisper_word_mapping',
        'dataset': 'yiddish_perfect_segments',
        'segments': processed_segments
    }
    
    metadata_file = output_path / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(segments_metadata, f, indent=2, ensure_ascii=False)
    
    # Get character analysis
    unique_chars = processor.get_unique_chars(all_texts)
    
    print(f"✅ Created TTS segments directory:")
    print(f"   Audio files: {len(processed_segments)}")
    print(f"   Text files: {len(processed_segments)}")
    print(f"   Total words: {sum(seg['word_count'] for seg in segments)}")
    print(f"   Unique characters: {len(unique_chars)}")
    print(f"   Characters: {''.join(unique_chars)}")
    
    return output_path, unique_chars


def create_training_file(segments, output_file="yiddish_train_data.txt"):
    """Create training file in TTS format"""
    
    print(f"📝 Creating TTS training file: {output_file}")
    
    processor = YiddishTextProcessor()
    
    # Create training data in format: audio_file|text|speaker_id
    training_data = []
    
    for segment in segments:
        # Use relative path from tts_segments
        rel_audio_path = f"tts_segments/{segment['audio_file']}"
        text = processor.normalize_yiddish_text(segment['text'])
        
        training_data.append(f"{rel_audio_path}|{text}|speaker_0")
    
    # Save training file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(training_data))
    
    print(f"💾 Training file saved: {output_file}")
    print(f"   Format: audio_file|text|speaker_id")
    print(f"   Entries: {len(training_data)}")
    
    return output_file


def main():
    """Main function"""
    
    print("🎯 PREPARING PERFECT SEGMENTS FOR TTS TRAINING")
    print("=" * 60)
    
    # Step 1: Collect perfect segments
    segments = collect_perfect_segments()
    if not segments:
        print("❌ No perfect segments found!")
        return
    
    # Step 2: Create TTS segments directory
    tts_dir, unique_chars = create_tts_segments_directory(segments)
    
    # Step 3: Create training file
    training_file = create_training_file(segments)
    
    # Step 4: Show next steps
    print(f"\n🎉 DATA PREPARATION COMPLETE!")
    print(f"✅ TTS segments: {tts_dir}")
    print(f"✅ Training file: {training_file}")
    print(f"✅ Total segments: {len(segments)}")
    print(f"✅ Unique characters: {len(unique_chars)}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run existing TTS training script:")
    print(f"   python3 prepare_yiddish_data.py")
    print(f"2. Or use TTS CLI:")
    print(f"   tts --model_name tts_models/en/ljspeech/tacotron2-DDC \\")
    print(f"       --train_data {training_file}")
    print(f"3. Your Yiddish TTS model will be ready! 🎤")


if __name__ == "__main__":
    main() 