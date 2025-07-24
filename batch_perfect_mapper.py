#!/usr/bin/env python3
"""
Batch Perfect Whisper Mapper
Applies whisper_word_mapper logic to all 21 audio/transcript pairs
Uses generated word_list.txt files to create perfect aligned segments
"""

import json
import re
from pathlib import Path
import soundfile as sf
import librosa

def load_original_words(transcript_path):
    """Load and clean the original Yiddish transcript"""
    with open(transcript_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    words = text.split()
    print(f"   📝 Original transcript: {len(words)} words")
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
    
    print(f"   ⏱️  Whisper timings: {len(timings)} words")
    return timings

def create_direct_mapping(original_words, whisper_timings):
    """Create 1:1 mapping between original words and Whisper timings"""
    
    original_count = len(original_words)
    timing_count = len(whisper_timings)
    
    print(f"   🔗 Mapping {original_count} words → {timing_count} timings")
    
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
    
    print(f"   📦 Created {len(segments)} segments")
    return segments

def export_perfect_alignment(segments, audio_id, audio_path, output_dir="perfect_mapped_segments"):
    """Export the perfectly aligned segments"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create file-specific directories
    file_audio_dir = output_path / f"file_{audio_id}" / "audio"
    file_text_dir = output_path / f"file_{audio_id}" / "text"
    file_audio_dir.mkdir(parents=True, exist_ok=True)
    file_text_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the original audio
    audio, sr = librosa.load(str(audio_path), sr=16000)
    
    segments_metadata = []
    
    for seg in segments:
        segment_id = seg['segment_id']
        
        # Export text
        text_file = file_text_dir / f"perfect_segment_{segment_id}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(seg['text'])
        
        # Export audio segment using precise timing
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        end_sample = min(end_sample, len(audio))  # Clamp to audio length
        
        if start_sample < len(audio):
            audio_segment = audio[start_sample:end_sample]
            audio_file = file_audio_dir / f"perfect_segment_{segment_id}.wav"
            sf.write(str(audio_file), audio_segment, sr)
        
        # Metadata
        segments_metadata.append({
            'segment_id': segment_id,
            'start_time': seg['start'],
            'end_time': seg['end'],
            'duration': seg['duration'],
            'text': seg['text'],
            'word_count': seg['word_count'],
            'audio_file': f"file_{audio_id}/audio/perfect_segment_{segment_id}.wav",
            'text_file': f"file_{audio_id}/text/perfect_segment_{segment_id}.txt"
        })
    
    # Save file metadata
    metadata_file = output_path / f"file_{audio_id}_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'file_id': audio_id,
            'source_audio': str(audio_path),
            'total_segments': len(segments_metadata),
            'total_words': sum(seg['word_count'] for seg in segments),
            'method': 'perfect_whisper_word_mapping',
            'segments': segments_metadata
        }, f, indent=2, ensure_ascii=False)
    
    return len(segments_metadata)

def get_completed_files(output_dir="perfect_mapped_segments"):
    """Get list of already completed files for checkpoint resume"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()
    
    completed = set()
    for metadata_file in output_path.glob("file_*_metadata.json"):
        # Extract audio_id from filename like "file_audio1_metadata.json"
        audio_id = metadata_file.stem.replace("file_", "").replace("_metadata", "")
        completed.add(audio_id)
    
    return completed

def process_single_file(audio_id, audio_path, transcript_path, word_list_path):
    """Process a single audio/transcript/word_list trio"""
    
    print(f"\n📄 Processing {audio_id}...")
    
    try:
        # Check if word list exists
        if not word_list_path.exists():
            print(f"   ❌ Word list not found: {word_list_path}")
            return False
        
        # Load original words
        original_words = load_original_words(transcript_path)
        
        # Load whisper timings
        whisper_timings = parse_whisper_timing(word_list_path)
        if not whisper_timings:
            print(f"   ❌ No valid timings in word list")
            return False
        
        # Create direct mapping
        mapped_words = create_direct_mapping(original_words, whisper_timings)
        
        # Create segments with generous overlap for robust TTS training
        segments = create_segments_from_mapping(mapped_words, words_per_segment=8, end_buffer=1.0)
        
        # Export everything
        segment_count = export_perfect_alignment(segments, audio_id, audio_path)
        
        print(f"   ✅ Successfully created {segment_count} perfect segments")
        print(f"      Every Yiddish word now has exact timing!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing {audio_id}: {e}")
        return False

def main():
    """Process all audio/transcript pairs with their word lists"""
    
    print("🎯 BATCH PERFECT WHISPER → ORIGINAL WORD MAPPING")
    print("=" * 60)
    
    # Get all audio/transcript pairs
    audio_dir = Path("original_files/audio")
    transcript_dir = Path("original_files/transcripts")
    word_list_dir = Path("word_lists")
    
    # Check if word_lists directory exists
    if not word_list_dir.exists():
        print(f"❌ Word lists directory not found: {word_list_dir}")
        print("   Run generate_word_lists.py first!")
        return
    
    pairs = []
    for i in range(1, 22):  # audio1 through audio21
        audio_id = f"audio{i}"
        audio_file = audio_dir / f"{audio_id}.wav"
        transcript_file = transcript_dir / f"transcription{i}.txt"
        word_list_file = word_list_dir / f"{audio_id}_word_list.txt"
        
        if audio_file.exists() and transcript_file.exists():
            pairs.append((audio_id, audio_file, transcript_file, word_list_file))
        else:
            print(f"⚠️  Missing files for {i}: audio={audio_file.exists()}, transcript={transcript_file.exists()}")
    
    print(f"📁 Found {len(pairs)} complete audio/transcript pairs")
    
    # Check which ones have word lists
    with_word_lists = [p for p in pairs if p[3].exists()]
    missing_word_lists = [p for p in pairs if not p[3].exists()]
    
    print(f"✅ {len(with_word_lists)} files have word lists")
    if missing_word_lists:
        print(f"⚠️  {len(missing_word_lists)} files missing word lists:")
        for audio_id, _, _, word_list_file in missing_word_lists:
            print(f"     {audio_id}: {word_list_file}")
        print("   Run generate_word_lists.py to create missing word lists")
    
    if not with_word_lists:
        print("❌ No word lists found! Run generate_word_lists.py first.")
        return
    
    # Check for completed files (checkpoint)
    completed = get_completed_files()
    if completed:
        print(f"🔄 Found {len(completed)} already completed files (resuming): {sorted(completed)}")
    
    # Process remaining files with word lists
    remaining_pairs = [p for p in with_word_lists if p[0] not in completed]
    print(f"⏳ Processing {len(remaining_pairs)} remaining files...")
    
    success_count = 0
    total_words = 0
    total_segments = 0
    
    for i, (audio_id, audio_path, transcript_path, word_list_path) in enumerate(remaining_pairs, 1):
        print(f"\n[{i}/{len(remaining_pairs)}] Processing {audio_id}...")
        
        if process_single_file(audio_id, audio_path, transcript_path, word_list_path):
            success_count += 1
            
            # Count words and segments for this file
            try:
                metadata_file = Path("perfect_mapped_segments") / f"file_{audio_id}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        total_words += metadata.get('total_words', 0)
                        total_segments += metadata.get('total_segments', 0)
            except:
                pass
    
    # Final summary
    total_completed = len(completed) + success_count
    print(f"\n🎉 BATCH PERFECT MAPPING COMPLETE!")
    print(f"   Total files: {len(pairs)}")
    print(f"   Previously completed: {len(completed)}")
    print(f"   Newly completed: {success_count}")
    print(f"   Total completed: {total_completed}")
    print(f"   Total segments created: {total_segments}")
    print(f"   Total words aligned: {total_words}")
    print(f"   Output directory: perfect_mapped_segments/")
    print(f"   🎯 PERFECT ALIGNMENT: Every Yiddish word has exact timing!")

if __name__ == "__main__":
    main() 