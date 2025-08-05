#!/usr/bin/env python3
"""
SIMPLE: Word-by-Word Whisper Timing
Just get word durations from Whisper and apply them 1:1 to original transcript
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def get_word_timings_from_whisper(audio_path):
    """Get word timings from Whisper - just durations and what was spoken"""
    import whisper
    
    print(f"🔄 Getting word timings from {Path(audio_path).name}...")
    
    # Load model
    model = whisper.load_model("base")
    
    # Transcribe with word timestamps (force Yiddish language)
    result = model.transcribe(audio_path, language='yi', word_timestamps=True)
    
    # Extract just the word timings
    word_timings = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                word_timings.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end'],
                    'duration': word['end'] - word['start']
                })
    
    print(f"   ✅ Got timings for {len(word_timings)} words")
    print(f"   Whisper detected: {' '.join([w['word'] for w in word_timings[:8]])}...")
    
    return word_timings

def align_words_simple(original_transcript, whisper_timings):
    """Simple 1:1 word alignment using Whisper timings"""
    print("🔗 Simple word-by-word alignment...")
    
    original_words = original_transcript.strip().split()
    
    print(f"   Original words: {len(original_words)}")
    print(f"   Whisper timings: {len(whisper_timings)}")
    print(f"   Original text: {' '.join(original_words[:8])}...")
    
    aligned_words = []
    
    # Use the shorter of the two lists to avoid index errors
    min_length = min(len(original_words), len(whisper_timings))
    
    for i in range(min_length):
        original_word = original_words[i]
        timing = whisper_timings[i]
        
        aligned_words.append({
            'word': original_word,  # Use original transcript word
            'start': timing['start'],
            'end': timing['end'],
            'duration': timing['duration'],
            'whisper_word': timing['word']  # What Whisper heard (for debugging)
        })
    
    # Handle extra original words if any
    if len(original_words) > len(whisper_timings):
        print(f"   ⚠️  Original has {len(original_words) - len(whisper_timings)} extra words")
        # Estimate timing for remaining words
        if aligned_words:
            last_end = aligned_words[-1]['end']
            avg_duration = sum(w['duration'] for w in aligned_words) / len(aligned_words)
            
            for i in range(len(whisper_timings), len(original_words)):
                start_time = last_end + (i - len(whisper_timings)) * avg_duration
                aligned_words.append({
                    'word': original_words[i],
                    'start': start_time,
                    'end': start_time + avg_duration,
                    'duration': avg_duration,
                    'whisper_word': '[estimated]'
                })
    
    print(f"   ✅ Aligned {len(aligned_words)} words")
    return aligned_words

def create_tts_segments_simple(aligned_words, target_duration=6):
    """Create TTS segments from aligned words"""
    print(f"✂️  Creating TTS segments (target: {target_duration}s per segment)")
    
    segments = []
    current_words = []
    current_start = None
    
    for i, word_info in enumerate(aligned_words):
        if current_start is None:
            current_start = word_info['start']
        
        current_words.append(word_info['word'])
        current_duration = word_info['end'] - current_start
        
        # End segment when we hit target duration or word limit
        should_end = (current_duration >= target_duration or 
                     len(current_words) >= 15 or 
                     i == len(aligned_words) - 1)
        
        if should_end:
            segment_text = ' '.join(current_words)
            segments.append({
                'start': current_start,
                'end': word_info['end'],
                'duration': word_info['end'] - current_start,
                'text': segment_text,
                'word_count': len(current_words)
            })
            
            print(f"   Segment {len(segments)}: {current_duration:.1f}s, {len(current_words)} words")
            print(f"     Text: {segment_text[:50]}...")
            
            current_words = []
            current_start = None
    
    print(f"   ✅ Created {len(segments)} segments")
    return segments

def extract_audio_segments_simple(audio_path, segments, output_dir="tts_segments_hebrew"):
    """Extract audio segments using simple timing"""
    print(f"🎵 Extracting segments to {output_dir}/")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    print(f"   Audio: {audio.shape[1]/sr:.1f}s total duration")
    
    extracted_segments = []
    
    for i, segment in enumerate(segments):
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        # Make sure we don't go beyond audio length
        end_sample = min(end_sample, audio.shape[1])
        
        if start_sample < audio.shape[1]:
            audio_segment = audio[:, start_sample:end_sample]
            
            # Save files
            segment_name = f"hebrew_segment_{i+1:04d}"
            audio_file = output_dir / "audio" / f"{segment_name}.wav"
            text_file = output_dir / "text" / f"{segment_name}.txt"
            
            torchaudio.save(str(audio_file), audio_segment, sr)
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(segment['text'])
            
            extracted_segments.append({
                'audio_file': str(audio_file),
                'text_file': str(text_file),
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'word_count': segment['word_count']
            })
            
            print(f"   ✅ {audio_file.name} - {segment['duration']:.1f}s")
    
    # Save metadata
    metadata_file = output_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_segments, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Metadata: {metadata_file}")
    return extracted_segments

def demo_simple_alignment():
    """Demo simple word-by-word alignment"""
    print("🔄 SIMPLE WORD-BY-WORD ALIGNMENT")
    print("=" * 50)
    print("Get word timings from Whisper → Apply to original transcript")
    print("=" * 50)
    
    # Find files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    
    if not audio_files or not transcript_files:
        print("❌ No files found in original_files/")
        return
    
    audio_path = str(audio_files[0])
    transcript_path = str(transcript_files[0])
    
    print(f"🎵 Audio: {Path(audio_path).name}")
    print(f"📝 Transcript: {Path(transcript_path).name}")
    
    try:
        # Load original transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            original_transcript = f.read().strip()
        
        print(f"\n📝 Original: {original_transcript[:100]}...")
        
        # Get word timings from Whisper
        word_timings = get_word_timings_from_whisper(audio_path)
        
        # Simple alignment
        aligned_words = align_words_simple(original_transcript, word_timings)
        
        # Create segments
        segments = create_tts_segments_simple(aligned_words)
        
        # Extract audio
        extracted = extract_audio_segments_simple(audio_path, segments)
        
        print(f"\n✅ SIMPLE ALIGNMENT COMPLETE!")
        print(f"   📁 Output: tts_segments_hebrew/")
        print(f"   🎵 Audio segments: {len(extracted)}")
        
        if extracted:
            print(f"\n🧪 First segment:")
            print(f"   Text: {extracted[0]['text']}")
            print(f"   Duration: {extracted[0]['duration']:.2f}s")
            print(f"   File: {extracted[0]['audio_file']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_simple_alignment() 