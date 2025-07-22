#!/usr/bin/env python3
"""
Manual Audio-Text Alignment Check
Create test segments and let user manually verify they match
"""

import os
import librosa
import torchaudio
import numpy as np
from pathlib import Path
import json

def create_test_segments(audio_path, transcript_path, num_segments=5, segment_length=6):
    """Create a few test segments for manual verification"""
    print(f"🧪 Creating {num_segments} test segments from {Path(audio_path).name}")
    
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read().strip()
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    print(f"   Original duration: {duration:.1f}s")
    print(f"   Text length: {len(transcript_text)} characters")
    
    # Create segments from the beginning (most likely to be aligned)
    segments = []
    total_chars_no_spaces = len(transcript_text.replace(' ', ''))
    
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min(start_time + segment_length, duration)
        
        if end_time <= start_time:
            break
        
        # Calculate what portion of text should be in this segment
        segment_start_ratio = start_time / duration
        segment_duration_ratio = (end_time - start_time) / duration
        
        # Find corresponding text
        char_start = int(segment_start_ratio * total_chars_no_spaces)
        char_count = int(segment_duration_ratio * total_chars_no_spaces)
        
        # Convert back to actual text positions (accounting for spaces)
        chars_found = 0
        text_start = 0
        
        for pos, char in enumerate(transcript_text):
            if char != ' ':
                if chars_found == char_start:
                    text_start = pos
                    break
                chars_found += 1
        
        # Find end position
        chars_in_segment = 0
        text_end = len(transcript_text)
        
        for pos in range(text_start, len(transcript_text)):
            if transcript_text[pos] != ' ':
                chars_in_segment += 1
                if chars_in_segment >= char_count:
                    # Try to end at word boundary
                    while pos < len(transcript_text) and transcript_text[pos] not in ' .,!?:':
                        pos += 1
                    text_end = pos
                    break
        
        segment_text = transcript_text[text_start:text_end].strip()
        
        segments.append({
            'segment_num': i + 1,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time,
            'text': segment_text,
            'word_count': len(segment_text.split()),
            'char_count': len(segment_text.replace(' ', ''))
        })
        
        print(f"   Segment {i+1}: {start_time:.1f}-{end_time:.1f}s ({len(segment_text.split())} words)")
    
    return segments, audio, sr

def extract_test_audio_files(segments, audio, sr, output_dir="alignment_test"):
    """Extract audio files for manual testing"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🎵 Extracting test audio files to {output_dir}/")
    
    # Convert to tensor for torchaudio
    import torch
    if len(audio.shape) == 1:
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add channel dimension
    else:
        audio_tensor = torch.FloatTensor(audio)
    
    extracted_files = []
    
    for segment in segments:
        # Extract audio segment
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio_tensor.shape[1] and end_sample <= audio_tensor.shape[1]:
            segment_audio = audio_tensor[:, start_sample:end_sample]
            
            # Save audio file
            audio_file = output_dir / f"test_segment_{segment['segment_num']:02d}.wav"
            text_file = output_dir / f"test_segment_{segment['segment_num']:02d}.txt"
            
            torchaudio.save(str(audio_file), segment_audio, sr)
            
            # Save text file
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(segment['text'])
            
            extracted_files.append({
                'audio_file': str(audio_file),
                'text_file': str(text_file),
                'segment': segment
            })
            
            print(f"   ✅ {audio_file.name} - {segment['duration']:.1f}s")
    
    return extracted_files

def create_verification_instructions(extracted_files, output_dir="alignment_test"):
    """Create instructions for manual verification"""
    output_dir = Path(output_dir)
    instructions_file = output_dir / "VERIFICATION_INSTRUCTIONS.txt"
    
    instructions = """
🔍 MANUAL ALIGNMENT VERIFICATION INSTRUCTIONS
=============================================

For each test segment, you need to verify that the audio matches the text.

HOW TO VERIFY:
1. Play the audio file
2. Read the corresponding text file
3. Check if what you hear matches what you read

WHAT TO LOOK FOR:
✅ GOOD: The spoken words match the written text (order and content)
❌ BAD: Different words, wrong order, or completely unrelated content

TEST SEGMENTS:
"""
    
    for i, file_info in enumerate(extracted_files):
        segment = file_info['segment']
        audio_name = Path(file_info['audio_file']).name
        
        instructions += f"""
Segment {i+1}: {audio_name}
   Time: {segment['start']:.1f}-{segment['end']:.1f}s ({segment['duration']:.1f}s)
   Text: {segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}
   Status: [ ] GOOD / [ ] BAD / [ ] UNSURE
   Notes: ________________________
"""
    
    instructions += """

RESULTS INTERPRETATION:
- If 4+ segments are GOOD: Segmentation method works well
- If 2-3 segments are GOOD: Segmentation needs some improvement  
- If 0-1 segments are GOOD: Segmentation method is not working

NEXT STEPS:
- If results are good: Proceed with full dataset segmentation
- If results are mixed: Try different segmentation parameters
- If results are bad: Need a completely different approach (forced alignment tools)
"""
    
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"📋 Instructions saved to: {instructions_file}")
    return str(instructions_file)

def manual_alignment_test():
    """Run manual alignment test"""
    print("🔍 Manual Audio-Text Alignment Test")
    print("=" * 40)
    
    # Find original files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))
    
    if not audio_files or not transcript_files:
        print("❌ No original files found in original_files/ directory")
        return
    
    # Use first file for testing
    audio_path = str(audio_files[0])
    transcript_path = str(transcript_files[0])
    
    print(f"Testing with: {Path(audio_path).name}")
    
    try:
        # Create test segments
        segments, audio, sr = create_test_segments(audio_path, transcript_path)
        
        if not segments:
            print("❌ Could not create test segments")
            return
        
        # Extract audio files
        extracted = extract_test_audio_files(segments, audio, sr)
        
        # Create instructions
        instructions_file = create_verification_instructions(extracted)
        
        print(f"\n✅ Test files created successfully!")
        print(f"\n📂 Files in alignment_test/:")
        test_dir = Path("alignment_test")
        for file in sorted(test_dir.glob("*")):
            print(f"   {file.name}")
        
        print(f"\n🔧 NEXT STEPS:")
        print(f"1. Open the alignment_test/ folder")
        print(f"2. Read: VERIFICATION_INSTRUCTIONS.txt")
        print(f"3. For each segment:")
        print(f"   - Play the .wav file")
        print(f"   - Read the .txt file") 
        print(f"   - Check if they match")
        print(f"4. Report back: How many segments matched correctly?")
        
        print(f"\n💡 TIP: Use any audio player (VLC, etc.) to play the .wav files")
        
    except Exception as e:
        print(f"❌ Error creating test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    manual_alignment_test() 