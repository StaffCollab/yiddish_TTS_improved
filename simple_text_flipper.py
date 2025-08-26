#!/usr/bin/env python3
"""
Simple text flipper - split text and reconstruct in opposite order
"""

def simple_flip_text(text):
    """
    Simple logic:
    1. Take original text
    2. Split (by spaces)
    3. Reconstruct in opposite order
    """
    words = text.split()
    words.reverse()
    return ' '.join(words)

def create_simple_flipped_dataset():
    """Create dataset with simple text flipping"""
    
    input_file = "ass_training_data_combined/yiddish_ass_precise_combined_train_data_106_ljspeech.txt"
    output_file = "ass_training_data_combined/yiddish_hebrew_simple_flip_106.txt"
    
    print(f"📖 Reading from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    flipped_lines = []
    
    print(f"🔄 Simple flipping {len(lines)} samples...")
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or '|' not in line:
            continue
            
        parts = line.split('|')
        if len(parts) < 2:
            continue
            
        audio_path = parts[0]
        hebrew_text = parts[1]
        
        # Fix audio path
        fixed_audio_path = f"wavs/{audio_path}.wav"
        
        # Simple flip: split by spaces, reverse order
        flipped_text = simple_flip_text(hebrew_text)
        
        # Create line
        flipped_line = f"{fixed_audio_path}|{flipped_text}|{flipped_text}"
        flipped_lines.append(flipped_line)
        
        if (i + 1) % 20 == 0:
            print(f"  ✅ Processed {i + 1}/{len(lines)} samples")
    
    # Write dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(flipped_lines))
    
    print(f"💾 Saved simple flipped dataset: {output_file}")
    
    # Show examples
    print(f"\n📝 Simple flip examples:")
    for i in range(min(5, len(flipped_lines))):
        parts = flipped_lines[i].split('|')
        # Get original for comparison
        orig_parts = lines[i].strip().split('|')
        
        print(f"  Sample {i+1}:")
        print(f"    Original: {orig_parts[1] if len(orig_parts) > 1 else 'N/A'}")
        print(f"    Flipped:  {parts[1]}")
        print()
    
    return output_file

if __name__ == "__main__":
    print("🎬 Simple Text Flipper")
    print("=" * 30)
    
    output_file = create_simple_flipped_dataset()
    
    print(f"✅ Created: {output_file}")
    print(f"💡 Update your FastSpeech2 config to use this dataset!")
    print(f"   Change meta_file_train to: yiddish_hebrew_simple_flip_106.txt") 