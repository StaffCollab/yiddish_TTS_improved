#!/usr/bin/env python3
"""
Alignment Diagnosis Tool
Compare Whisper transcription vs Original transcript to find mismatches
"""

import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def diagnose_transcript_mismatch():
    """Compare what Whisper actually hears vs original transcript"""
    print("🔍 ALIGNMENT DIAGNOSIS")
    print("=" * 50)
    print("Comparing Whisper transcription vs Original transcript")
    print("=" * 50)
    
    # Find files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    
    if not audio_files or not transcript_files:
        print("❌ No original files found")
        return
    
    audio_path = str(audio_files[0])
    transcript_path = str(transcript_files[0])
    
    print(f"📁 Audio: {Path(audio_path).name}")
    print(f"📁 Transcript: {Path(transcript_path).name}")
    
    # Load original transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        original_transcript = f.read().strip()
    
    original_words = original_transcript.split()
    print(f"\n📝 Original transcript: {len(original_words)} words")
    print(f"   First 10 words: {' '.join(original_words[:10])}")
    
    # Get Whisper transcription
    print(f"\n🎤 Running Whisper transcription...")
    
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Extract Whisper words and timings
    whisper_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                whisper_words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
    
    whisper_text_only = [w['word'] for w in whisper_words]
    
    print(f"   ✅ Whisper heard: {len(whisper_words)} words")
    print(f"   First 10 words: {' '.join(whisper_text_only[:10])}")
    
    # Compare first 20 words in detail
    print(f"\n🔍 DETAILED COMPARISON (First 20 words):")
    print(f"{'#':<3} {'Original':<30} {'Whisper':<30} {'Match':<8} {'Timing'}")
    print("=" * 80)
    
    mismatches = 0
    for i in range(min(20, len(original_words), len(whisper_text_only))):
        orig_word = original_words[i]
        whisper_word = whisper_text_only[i]
        match = "✅" if orig_word.lower() == whisper_word.lower() else "❌"
        
        if i < len(whisper_words):
            timing = f"{whisper_words[i]['start']:.1f}-{whisper_words[i]['end']:.1f}s"
        else:
            timing = "N/A"
        
        print(f"{i+1:<3} {orig_word:<30} {whisper_word:<30} {match:<8} {timing}")
        
        if orig_word.lower() != whisper_word.lower():
            mismatches += 1
    
    print("=" * 80)
    print(f"📊 Mismatches in first 20 words: {mismatches}/20 ({mismatches/20*100:.1f}%)")
    
    # Overall statistics
    total_mismatches = 0
    max_compare = min(len(original_words), len(whisper_text_only))
    
    for i in range(max_compare):
        if original_words[i].lower() != whisper_text_only[i].lower():
            total_mismatches += 1
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Original words: {len(original_words)}")
    print(f"   Whisper words: {len(whisper_text_only)}")
    print(f"   Comparable words: {max_compare}")
    print(f"   Total mismatches: {total_mismatches}/{max_compare} ({total_mismatches/max_compare*100:.1f}%)")
    
    # Show complete Whisper transcription vs Original
    print(f"\n📄 COMPLETE WHISPER TRANSCRIPTION:")
    whisper_full_text = ' '.join(whisper_text_only)
    print(f"   {whisper_full_text[:200]}...")
    
    print(f"\n📄 ORIGINAL TRANSCRIPT:")
    print(f"   {original_transcript[:200]}...")
    
    # Diagnosis
    print(f"\n🎯 DIAGNOSIS:")
    
    if total_mismatches / max_compare > 0.5:
        print("❌ MAJOR MISMATCH: Original transcript differs significantly from audio")
        print("   💡 SOLUTION: Use Whisper's transcription instead of original")
        print("   🔧 This will create properly aligned segments")
    elif total_mismatches / max_compare > 0.2:
        print("⚠️  MODERATE MISMATCH: Some differences between transcript and audio")
        print("   💡 SOLUTION: Consider using Whisper transcription or hybrid approach")
    else:
        print("✅ GOOD MATCH: Transcript generally matches audio")
        print("   💡 ISSUE: May be timing precision problems")
    
    print(f"\n🔧 RECOMMENDED ACTION:")
    if total_mismatches / max_compare > 0.3:
        print("1. Use Whisper's transcription as the source text")
        print("2. Create segments from Whisper's word-level timings")
        print("3. This guarantees perfect audio-text alignment")
    else:
        print("1. Fine-tune the timing alignment algorithm")
        print("2. Add more sophisticated word matching")
    
    return {
        'original_words': original_words,
        'whisper_words': whisper_words,
        'mismatch_rate': total_mismatches / max_compare if max_compare > 0 else 1.0
    }

if __name__ == "__main__":
    diagnose_transcript_mismatch() 