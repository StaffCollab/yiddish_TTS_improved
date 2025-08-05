#!/usr/bin/env python3
"""
Fix Yiddish TTS Training Data Issues
Addresses speaker ID format, vocabulary mismatch, and short audio segments
"""

import os
import json
import glob
from pathlib import Path
import librosa
import soundfile as sf
from collections import Counter

def analyze_current_data():
    """Analyze current training data for issues"""
    print("🔍 ANALYZING CURRENT TRAINING DATA")
    print("=" * 50)
    
    # Read current training file
    with open("yiddish_tacotron2_train_data.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"📊 Total training samples: {len(lines)}")
    
    # Analyze format issues
    format_issues = 0
    character_issues = []
    
    for i, line in enumerate(lines[:10]):  # Check first 10
        parts = line.strip().split('|')
        if len(parts) == 3:
            audio_path, text, speaker = parts
            print(f"Line {i+1}: {audio_path.split('/')[-1]} | {text[:30]}... | {speaker}")
            
            # Check for ASCII characters in speaker field
            if speaker == "yiddish_speaker":
                format_issues += 1
                
            # Check for non-Hebrew characters in text
            for char in text:
                if ord(char) >= 65 and ord(char) <= 122:  # ASCII letters
                    character_issues.append(char)
    
    print(f"\n⚠️  Format issues found: {format_issues}")
    print(f"⚠️  ASCII characters in text: {set(character_issues)}")
    
    return lines

def check_audio_durations():
    """Check audio segment durations"""
    print("\n🎵 CHECKING AUDIO DURATIONS")
    print("=" * 50)
    
    durations = []
    short_files = []
    
    metadata_files = glob.glob("perfect_mapped_segments/file_*_metadata.json")
    
    for metadata_file in metadata_files[:3]:  # Check first 3 for speed
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for segment in metadata['segments']:
            duration = segment['duration']
            durations.append(duration)
            
            if duration < 0.5:  # Flag very short segments
                short_files.append((segment['audio_file'], duration))
    
    print(f"📊 Analyzed {len(durations)} segments")
    print(f"📏 Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"📈 Average duration: {sum(durations)/len(durations):.2f}s")
    print(f"⚠️  Short segments (< 0.5s): {len(short_files)}")
    
    if short_files:
        print("   Short segments:")
        for file, dur in short_files[:5]:
            print(f"     {file}: {dur:.2f}s")
    
    return durations, short_files

def collect_all_characters():
    """Collect all unique characters from training text"""
    print("\n🔤 COLLECTING CHARACTER VOCABULARY")
    print("=" * 50)
    
    all_chars = set()
    
    # Read from metadata files to get clean text
    metadata_files = glob.glob("perfect_mapped_segments/file_*_metadata.json")
    
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for segment in metadata['segments']:
            text = segment['text']
            all_chars.update(text)
    
    # Separate Hebrew letters, punctuation, and other characters
    hebrew_chars = []
    punctuation = []
    numbers = []
    spaces = []
    other_chars = []
    
    for char in sorted(all_chars):
        if 0x0590 <= ord(char) <= 0x05FF or 0xFB1D <= ord(char) <= 0xFB4F:
            hebrew_chars.append(char)
        elif char in ".,!?;:-()[]{}\"'`":
            punctuation.append(char)
        elif char.isdigit():
            numbers.append(char)
        elif char in " \n\t":
            spaces.append(char)
        else:
            other_chars.append(char)
    
    print(f"📚 Character Analysis:")
    print(f"   Hebrew letters: {len(hebrew_chars)} - {''.join(hebrew_chars[:20])}...")
    print(f"   Punctuation: {len(punctuation)} - {''.join(punctuation)}")
    print(f"   Numbers: {len(numbers)} - {''.join(numbers)}")
    print(f"   Spaces: {len(spaces)} - {repr(''.join(spaces))}")
    print(f"   Other: {len(other_chars)} - {''.join(other_chars)}")
    
    return {
        'hebrew': ''.join(hebrew_chars),
        'punctuation': ''.join(punctuation),
        'numbers': ''.join(numbers),
        'spaces': ''.join(spaces),
        'other': ''.join(other_chars)
    }

def create_fixed_training_file(min_duration=0.5):
    """Create properly formatted training file"""
    print(f"\n🔧 CREATING FIXED TRAINING FILE")
    print("=" * 50)
    
    new_training_data = []
    filtered_count = 0
    
    metadata_files = glob.glob("perfect_mapped_segments/file_*_metadata.json")
    
    for metadata_file in metadata_files:
        print(f"   Processing {Path(metadata_file).name}...")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for segment in metadata['segments']:
            # Filter by duration
            if segment['duration'] >= min_duration:
                # Proper format: audio_path|text|speaker_id
                audio_path = f"perfect_mapped_segments/{segment['audio_file']}"
                text = segment['text'].strip()
                speaker_id = "speaker_0"  # Single speaker
                
                new_training_data.append(f"{audio_path}|{text}|{speaker_id}")
            else:
                filtered_count += 1
    
    # Save fixed training file
    output_file = "yiddish_tacotron2_train_data_FIXED.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_training_data))
    
    print(f"✅ Fixed training file created: {output_file}")
    print(f"📊 Total samples: {len(new_training_data)}")
    print(f"🗑️  Filtered short segments: {filtered_count}")
    
    return output_file

def create_updated_config(characters_dict, training_file):
    """Create updated training configuration"""
    print(f"\n⚙️  CREATING UPDATED CONFIG")
    print("=" * 50)
    
    # Separate characters from punctuation to avoid duplicates
    # Only Hebrew letters, numbers, spaces, and other chars (NOT punctuation)
    non_punctuation_chars = (
        characters_dict['hebrew'] + 
        characters_dict['numbers'] + 
        characters_dict['spaces'] + 
        characters_dict['other']
    )
    
    config_updates = {
        "characters": {
            "pad": "_",
            "eos": "~", 
            "bos": "^",
            "characters": non_punctuation_chars,  # NO punctuation here
            "punctuations": characters_dict['punctuation'],  # Only punctuation here
            "phonemes": "",
            "is_unique": True,
            "is_sorted": True
        },
        "datasets": [
            {
                "name": "yiddish_perfect_fixed",
                "path": "./",
                "meta_file_train": training_file,
                "meta_file_val": training_file,
                "formatter": "ljspeech"
            }
        ],
        "test_sentences": [
            "געווען איז דאס פאריגע וואך מיטוואך, ווען איך",
            "האב - אינאיינעם מיט נאך עטליכע - פארוויילט",
            "אין אמעריקע'ס אפיציעלע הויפטשטאט - וואשינגטאן די-סי. נישט",
            "צו שטיין און שמועסן, אבער אזויפיל האט ער",
            "שבת שלום און א גוטן טאג"
        ],
        "min_audio_len": 0.5,  # Filter short segments
        "max_audio_len": 20.0,  # Reasonable max
        "min_text_len": 5,
        "max_text_len": 500,
        "batch_size": 4,  # Smaller for stability
        "lr": 5e-4,  # Lower learning rate
        "print_step": 25,
        "save_step": 500,
        "epochs": 100,
        "run_eval": True,
        "test_delay_epochs": 5
    }
    
    # Save config updates
    with open("yiddish_config_FIXED.json", 'w', encoding='utf-8') as f:
        json.dump(config_updates, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated config saved: yiddish_config_FIXED.json")
    print(f"📝 Non-punctuation characters: {len(non_punctuation_chars)}")
    print(f"📝 Punctuation characters: {len(characters_dict['punctuation'])}")
    print(f"🔤 Character set: {non_punctuation_chars[:30]}...")
    print(f"🔤 Punctuation set: {characters_dict['punctuation']}")
    
    return config_updates

def main():
    """Run complete data analysis and fixes"""
    print("🎯 YIDDISH TTS TRAINING DATA FIXER")
    print("=" * 60)
    print("Fixing: Speaker ID format, vocabulary mismatch, short segments")
    print("=" * 60)
    
    # Step 1: Analyze current issues
    current_data = analyze_current_data()
    
    # Step 2: Check audio durations
    durations, short_files = check_audio_durations()
    
    # Step 3: Collect proper character vocabulary
    characters_dict = collect_all_characters()
    
    # Step 4: Create fixed training file
    fixed_training_file = create_fixed_training_file(min_duration=0.5)
    
    # Step 5: Create updated config
    config_updates = create_updated_config(characters_dict, fixed_training_file)
    
    print(f"\n🎉 FIXES COMPLETED!")
    print("=" * 60)
    print("✅ Created fixed training file with proper speaker ID format")
    print("✅ Filtered out segments shorter than 0.5 seconds")
    print("✅ Built complete character vocabulary (Hebrew + punctuation)")
    print("✅ Created updated config with lower learning rate")
    print("✅ Added proper Yiddish test sentences")
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review: yiddish_tacotron2_train_data_FIXED.txt")
    print(f"2. Review: yiddish_config_FIXED.json")
    print(f"3. Update your training script to use these files")
    print(f"4. Restart training with fixed configuration")

if __name__ == "__main__":
    main() 