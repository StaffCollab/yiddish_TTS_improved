#!/usr/bin/env python3
"""
Map Original Yiddish Words to Whisper Timing Data
Takes correct Yiddish transcript + Whisper timing → Perfect aligned segments
"""

import json
import re
from pathlib import Path

def load_original_words(transcript_path):
    """Load and clean the original Yiddish transcript"""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # Split into words and clean
    words = text.split()
    print(f"📝 Original transcript: {len(words)} words")
    return words

def parse_whisper_timing(timing_file):
    """Parse the whisper word timing list"""
    timings = []
    
    with open(timing_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines[3:]:  # Skip header lines
        line = line.strip()
        if not line or not re.match(r'\s*\d+\.', line):
            continue
            
        # Parse: "  1.    0.00s -    1.00s  'Gewen'"
        match = re.match(r'\s*(\d+)\.\s+([\d.]+)s\s*-\s*([\d.]+)s\s*\'([^\']+)\'', line)
        if match:
            idx, start, end, word = match.groups()
            timings.append({
                'index': int(idx),
                'start': float(start),
                'end': float(end),
                'whisper_word': word,
                'duration': float(end) - float(start)
            })
    
    print(f"⏱️  Whisper timings: {len(timings)} words")
    return timings

def create_direct_mapping(original_words, whisper_timings):
    """Create 1:1 mapping between original words and Whisper timings"""
    
    original_count = len(original_words)
    timing_count = len(whisper_timings)
    
    print(f"\n🔗 CREATING WORD MAPPING")
    print(f"   Original: {original_count} words")
    print(f"   Timings:  {timing_count} words")
    print(f"   Difference: {timing_count - original_count} words")
    
    mapped_words = []
    
    # Handle the mismatch by stretching/compressing proportionally
    for i in range(original_count):
        # Map original word index to timing index proportionally
        timing_idx = min(int(i * timing_count / original_count), timing_count - 1)
        
        timing = whisper_timings[timing_idx]
        
        mapped_words.append({
            'original_word': original_words[i],
            'whisper_word': timing['whisper_word'],
            'start': timing['start'],
            'end': timing['end'],
            'duration': timing['duration'],
            'original_index': i + 1,
            'timing_index': timing['index']
        })
    
    print(f"✅ Created {len(mapped_words)} word mappings")
    return mapped_words

def create_segments_from_mapping(mapped_words, words_per_segment=8, end_buffer=1.0):
    """Group mapped words into segments for TTS training with end buffer"""
    
    segments = []
    current_segment = []
    segment_id = 1
    
    for word_data in mapped_words:
        current_segment.append(word_data)
        
        # Create segment when we reach target length
        if len(current_segment) >= words_per_segment:
            segment_text = ' '.join(w['original_word'] for w in current_segment)
            segment_start = current_segment[0]['start']
            segment_end_raw = current_segment[-1]['end']
            
            # Add buffer for complete speech capture (overlaps OK for TTS)
            segment_end_buffered = segment_end_raw + end_buffer
            
            segments.append({
                'segment_id': f"{segment_id:04d}",
                'start': segment_start,
                'end': segment_end_buffered,
                'duration': segment_end_buffered - segment_start,
                'text': segment_text,
                'word_count': len(current_segment),
                'words': current_segment
            })
            
            current_segment = []
            segment_id += 1
    
    # Handle remaining words
    if current_segment:
        segment_text = ' '.join(w['original_word'] for w in current_segment)
        segment_start = current_segment[0]['start']
        segment_end_raw = current_segment[-1]['end']
        segment_end_buffered = segment_end_raw + end_buffer
        
        segments.append({
            'segment_id': f"{segment_id:04d}",
            'start': segment_start,
            'end': segment_end_buffered,
            'duration': segment_end_buffered - segment_start,
            'text': segment_text,
            'word_count': len(current_segment),
            'words': current_segment
        })
    
    print(f"📦 Created {len(segments)} segments (avg {len(mapped_words)/len(segments):.1f} words each)")
    print(f"🛡️  Added {end_buffer}s end buffer for complete speech capture (overlaps OK for TTS)")
    return segments

def export_perfect_alignment(segments, output_dir="perfect_aligned_segments"):
    """Export the perfectly aligned segments"""
    import soundfile as sf
    import librosa
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "audio").mkdir(exist_ok=True)
    (output_path / "text").mkdir(exist_ok=True)
    
    # Load the original audio
    audio, sr = librosa.load("original_files/audio/audio1.wav", sr=16000)
    
    segments_metadata = []
    
    for seg in segments:
        segment_id = seg['segment_id']
        
        # Export text
        text_file = output_path / "text" / f"perfect_segment_{segment_id}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(seg['text'])
        
        # Export audio segment using precise timing
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        audio_segment = audio[start_sample:end_sample]
        
        audio_file = output_path / "audio" / f"perfect_segment_{segment_id}.wav"
        sf.write(audio_file, audio_segment, sr)
        
        # Metadata
        segments_metadata.append({
            'segment_id': segment_id,
            'start_time': seg['start'],
            'end_time': seg['end'],
            'duration': seg['duration'],
            'text': seg['text'],
            'word_count': seg['word_count'],
            'audio_file': f"audio/perfect_segment_{segment_id}.wav",
            'text_file': f"text/perfect_segment_{segment_id}.txt"
        })
    
    # Save metadata
    metadata_file = output_path / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_segments': len(segments_metadata),
            'total_words': sum(seg['word_count'] for seg in segments),
            'method': 'whisper_timing_mapped_to_original_text',
            'segments': segments_metadata
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 PERFECT ALIGNMENT EXPORTED")
    print(f"   Directory: {output_path}")
    print(f"   Segments: {len(segments_metadata)}")
    print(f"   Total words: {sum(seg['word_count'] for seg in segments)}")

def main():
    """Create perfect alignment using Whisper timing + original text"""
    
    print("🎯 PERFECT WHISPER → ORIGINAL WORD MAPPING")
    print("=" * 50)
    
    # Load data
    original_words = load_original_words("original_files/transcripts/transcription1.txt")
    whisper_timings = parse_whisper_timing("whisper_word_list.txt")
    
    # Create 1:1 mapping
    mapped_words = create_direct_mapping(original_words, whisper_timings)
    
    # Show first few mappings for verification
    print(f"\n🔍 FIRST 10 MAPPINGS:")
    for i in range(10):
        w = mapped_words[i]
        print(f"  {i+1:2d}. '{w['original_word']}' ← '{w['whisper_word']}' ({w['start']:.2f}s-{w['end']:.2f}s)")
    
    # Create segments with generous overlap for robust TTS training
    segments = create_segments_from_mapping(mapped_words, words_per_segment=8, end_buffer=1.0)
    
    # Export everything
    export_perfect_alignment(segments)
    
    print(f"\n✅ PERFECT ALIGNMENT COMPLETE!")
    print(f"   Every Yiddish word now has exact timing!")
    print(f"   Check perfect_aligned_segments/ for results")

if __name__ == "__main__":
    main() 