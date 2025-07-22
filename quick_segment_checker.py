#!/usr/bin/env python3
"""
Quick Segment Alignment Checker
Test a few existing segments to see if any are properly aligned
"""

import json
import random
from pathlib import Path
import torchaudio
import torch

def check_existing_segments():
    """Check if existing TTS segments have reasonable alignment"""
    print("🔍 Quick Check: Are Existing TTS Segments Usable?")
    print("=" * 50)
    
    # Load metadata
    with open('tts_segments/segments_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} existing segments")
    
    # Analyze segment characteristics
    durations = [item['duration'] for item in metadata]
    text_lengths = [len(item['text']) for item in metadata]
    
    avg_duration = sum(durations) / len(durations)
    avg_text_length = sum(text_lengths) / len(text_lengths)
    
    print(f"\n📊 Segment Analysis:")
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   Average text length: {avg_text_length:.0f} characters")
    
    # Check duration distribution
    good_duration = sum(1 for d in durations if 2 <= d <= 10)
    print(f"   TTS-suitable durations: {good_duration}/{len(durations)} ({good_duration/len(durations)*100:.1f}%)")
    
    # Sample a few segments for manual inspection
    print(f"\n🧪 Sample Segments for Manual Testing:")
    
    # Get 5 random segments
    sample_indices = random.sample(range(len(metadata)), min(5, len(metadata)))
    
    test_dir = Path("quick_alignment_test")
    test_dir.mkdir(exist_ok=True)
    
    print(f"   Creating test files in {test_dir}/")
    
    for i, idx in enumerate(sample_indices):
        segment = metadata[idx]
        
        # Copy audio and text files for easy testing
        audio_file = Path(segment['audio_file'])
        text_content = segment['text']
        
        if audio_file.exists():
            # Copy audio file
            test_audio = test_dir / f"test_{i+1:02d}_{audio_file.name}"
            test_text = test_dir / f"test_{i+1:02d}_text.txt"
            
            # Copy audio
            import shutil
            shutil.copy2(audio_file, test_audio)
            
            # Save text
            with open(test_text, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"\n   Test {i+1}:")
            print(f"      Audio: {test_audio.name}")
            print(f"      Duration: {segment['duration']:.1f}s")
            print(f"      Text: {text_content[:60]}...")
            print(f"      Characters: {len(text_content)}")
            
            # Calculate speech rate
            word_count = len(text_content.split())
            wpm = (word_count / segment['duration']) * 60
            cps = len(text_content.replace(' ', '')) / segment['duration']
            
            print(f"      Rate: {wpm:.0f} WPM, {cps:.1f} chars/sec")
            
            # Quality assessment
            if 100 <= wpm <= 250 and 2 <= segment['duration'] <= 10:
                quality = "✅ GOOD"
            elif 80 <= wpm <= 300 and 1 <= segment['duration'] <= 12:
                quality = "⚠️  FAIR"
            else:
                quality = "❌ POOR"
            
            print(f"      Quality: {quality}")
    
    # Create simple test instructions
    instructions = """
🔍 QUICK ALIGNMENT TEST INSTRUCTIONS
====================================

For each test file:
1. Play the audio file (test_XX_segment_XXXX.wav)
2. Read the text file (test_XX_text.txt) 
3. Check: Does the audio match the text?

WHAT TO EXPECT:
✅ GOOD: Audio contains exactly the text shown
⚠️  PARTIAL: Audio contains some of the text but not all
❌ BAD: Audio is completely different from text

If 2+ segments are GOOD → existing segments might be usable!
If 0-1 segments are GOOD → need better alignment method
"""
    
    with open(test_dir / "TEST_INSTRUCTIONS.txt", 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print(f"\n📋 Instructions saved: {test_dir}/TEST_INSTRUCTIONS.txt")
    
    # Overall assessment
    print(f"\n🎯 QUICK ASSESSMENT:")
    if avg_duration <= 10 and good_duration/len(durations) > 0.7:
        print("✅ Existing segments have reasonable durations")
        print("   → Worth testing for alignment quality")
    else:
        print("⚠️  Existing segments have duration issues")
        print("   → May need re-segmentation regardless")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Test the sample segments in quick_alignment_test/")
    print("2. If alignment is good → use existing segments!")
    print("3. If alignment is poor → try Whisper alignment")
    print("4. Report back: How many test segments matched?")

if __name__ == "__main__":
    check_existing_segments() 