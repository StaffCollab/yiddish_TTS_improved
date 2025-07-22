#!/usr/bin/env python3
"""
Manual Split Helper
Assists with manually creating audio-text segments
"""

import os
import json
from pathlib import Path

def setup_manual_split_workspace(output_dir="manual_segments"):
    """Set up workspace for manual splitting"""
    print("🔧 MANUAL SPLIT WORKSPACE SETUP")
    print("=" * 40)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Copy original transcript for reference
    transcript_path = "original_files/transcripts/transcription1.txt"
    if Path(transcript_path).exists():
        with open(transcript_path, 'r', encoding='utf-8') as f:
            original_text = f.read().strip()
        
        # Save reference transcript
        with open(output_dir / "original_transcript.txt", 'w', encoding='utf-8') as f:
            f.write(original_text)
        
        # Create template for splitting
        words = original_text.split()
        
        # Create a template showing suggested splits
        template_content = []
        template_content.append("# MANUAL SPLITTING TEMPLATE")
        template_content.append("# Copy the text for each audio segment you create")
        template_content.append("# Aim for 5-15 words per segment\n")
        
        current_chunk = []
        chunk_num = 1
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            # Suggest split every ~10 words
            if len(current_chunk) >= 10 or i == len(words) - 1:
                text_chunk = ' '.join(current_chunk)
                template_content.append(f"=== SEGMENT {chunk_num:02d} ===")
                template_content.append(text_chunk)
                template_content.append("")
                
                current_chunk = []
                chunk_num += 1
        
        with open(output_dir / "splitting_template.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(template_content))
        
        print(f"✅ Workspace created: {output_dir}/")
        print(f"📄 Original transcript: {output_dir}/original_transcript.txt")
        print(f"📋 Splitting template: {output_dir}/splitting_template.txt")
        
        return output_dir
    else:
        print("❌ Original transcript not found!")
        return None

def create_text_segment(segment_number, text_content, output_dir="manual_segments"):
    """Create a text file for a manual segment"""
    output_dir = Path(output_dir)
    text_file = output_dir / "text" / f"manual_segment_{segment_number:04d}.txt"
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_content.strip())
    
    print(f"✅ Created: {text_file}")
    return text_file

def validate_manual_segments(output_dir="manual_segments"):
    """Check manual segments and create metadata"""
    print("🔍 VALIDATING MANUAL SEGMENTS")
    print("=" * 35)
    
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    text_dir = output_dir / "text"
    
    audio_files = sorted(audio_dir.glob("manual_segment_*.wav"))
    text_files = sorted(text_dir.glob("manual_segment_*.txt"))
    
    print(f"Audio files: {len(audio_files)}")
    print(f"Text files: {len(text_files)}")
    
    if len(audio_files) != len(text_files):
        print(f"⚠️  Mismatch: {len(audio_files)} audio vs {len(text_files)} text files")
    
    # Create metadata for existing pairs
    segments_metadata = []
    
    for i, (audio_file, text_file) in enumerate(zip(audio_files, text_files)):
        with open(text_file, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
        
        segments_metadata.append({
            'segment_number': i + 1,
            'audio_file': str(audio_file),
            'text_file': str(text_file),
            'text': text_content,
            'word_count': len(text_content.split())
        })
        
        print(f"✅ Segment {i+1}: {len(text_content.split())} words")
        print(f"   Text: {text_content[:50]}...")
    
    # Save metadata
    metadata_file = output_dir / "manual_segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(segments_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Metadata saved: {metadata_file}")
    return segments_metadata

def print_manual_instructions():
    """Print step-by-step manual splitting instructions"""
    print("📖 MANUAL SPLITTING INSTRUCTIONS")
    print("=" * 40)
    print()
    print("🎧 STEP 1: Install Audacity (free)")
    print("   Download from: https://www.audacityteam.org/")
    print()
    print("🎵 STEP 2: Open your audio file")
    print("   File → Open → original_files/audio/audio1.wav")
    print()
    print("✂️  STEP 3: Split the audio")
    print("   1. Listen and find natural speech pauses")
    print("   2. Select each segment (click & drag)")
    print("   3. File → Export → Export Selected Audio")
    print("   4. Save as: manual_segments/audio/manual_segment_0001.wav")
    print("   5. Repeat for each segment (0002, 0003, etc.)")
    print()
    print("📝 STEP 4: Create matching text files")
    print("   1. Use the splitting_template.txt as reference")
    print("   2. For each audio segment, save the matching text as:")
    print("      manual_segments/text/manual_segment_0001.txt")
    print()
    print("✅ STEP 5: Validate your work")
    print("   Run: python manual_split_helper.py validate")
    print()
    print("🎯 TIPS:")
    print("   • Aim for 5-15 words per segment")
    print("   • Split at natural speech pauses")
    print("   • Keep segments 3-10 seconds long")
    print("   • Number segments consistently (0001, 0002, etc.)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_manual_segments()
    elif len(sys.argv) > 1 and sys.argv[1] == "instructions":
        print_manual_instructions()
    else:
        workspace = setup_manual_split_workspace()
        if workspace:
            print()
            print_manual_instructions()
            print(f"\n🚀 Ready to start! Your workspace is: {workspace}/") 