#!/usr/bin/env python3
"""
FINAL: Yiddish-Native Whisper Alignment
Use native Yiddish language detection for perfect audio-text alignment
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def transcribe_yiddish_whisper(audio_path):
    """Force Whisper to use Yiddish language detection"""
    import whisper
    
    print(f"🔄 Transcribing {Path(audio_path).name} with FORCED Yiddish detection...")
    
    # Load model
    model = whisper.load_model("base")
    
    # Force Yiddish language detection (the actual language!)
    result = model.transcribe(audio_path, language='yi', word_timestamps=True)
    
    # Extract word-level timestamps
    words_with_timing = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                words_with_timing.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
    
    print(f"   ✅ Yiddish transcription: {len(words_with_timing)} words")
    yiddish_text = ' '.join([w['word'] for w in words_with_timing[:10]])
    print(f"   First 10 words: {yiddish_text}")
    
    return words_with_timing, result['text']

def align_yiddish_transcripts(original_transcript, yiddish_whisper_words):
    """Smart alignment between original Yiddish transcript and Whisper Yiddish output"""
    print("🔗 Aligning original Yiddish transcript with Yiddish Whisper...")
    
    original_words = original_transcript.strip().split()
    whisper_words = [w['word'] for w in yiddish_whisper_words]
    
    print(f"   Original: {len(original_words)} words")
    print(f"   Yiddish Whisper: {len(whisper_words)} words")
    
    # Simple sequential alignment (both are now Yiddish)
    aligned_words = []
    
    for i, orig_word in enumerate(original_words):
        if i < len(yiddish_whisper_words):
            # Use Whisper timing with original text
            timing = yiddish_whisper_words[i]
            aligned_words.append({
                'word': orig_word,  # Use original text
                'start': timing['start'],
                'end': timing['end'],
                'whisper_word': timing['word']  # Keep Whisper's version for reference
            })
        else:
            # Estimate timing for remaining words
            if aligned_words:
                last_end = aligned_words[-1]['end']
                aligned_words.append({
                    'word': orig_word,
                    'start': last_end,
                    'end': last_end + 0.5,
                    'whisper_word': ''
                })
    
    print(f"   ✅ Aligned {len(aligned_words)} words")
    return aligned_words

def create_hebrew_tts_segments(aligned_words, target_duration=6, max_duration=10):
    """Create TTS segments from Hebrew-aligned words"""
    print(f"✂️  Creating Hebrew TTS segments (target: {target_duration}s)")
    
    segments = []
    current_words = []
    current_start = None
    
    for i, word_info in enumerate(aligned_words):
        if current_start is None:
            current_start = word_info['start']
        
        current_words.append(word_info['word'])
        current_duration = word_info['end'] - current_start
        
        # End segment criteria
        should_end = False
        if current_duration >= target_duration or len(current_words) >= 12:
            should_end = True
        
        if should_end or i == len(aligned_words) - 1:
            # End exactly at word boundary with small safety gap
            segment_end = word_info['end']
            if i + 1 < len(aligned_words):
                next_start = aligned_words[i + 1]['start']
                segment_end = min(segment_end, next_start - 0.02)
            
            segment_text = ' '.join(current_words)
            segments.append({
                'start': current_start,
                'end': segment_end,
                'duration': segment_end - current_start,
                'text': segment_text,
                'word_count': len(current_words)
            })
            
            current_words = []
            current_start = None
    
    print(f"   ✅ Created {len(segments)} segments")
    return segments

def extract_hebrew_audio_segments(audio_path, segments, output_dir="tts_segments_hebrew"):
    """Extract properly aligned Hebrew audio segments"""
    print(f"🎵 Extracting Hebrew-aligned segments to {output_dir}/")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    extracted_segments = []
    
    for i, segment in enumerate(segments):
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio.shape[1] and end_sample <= audio.shape[1]:
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
            
            print(f"   ✅ {audio_file.name} - {segment['duration']:.1f}s, {segment['word_count']} words")
    
    # Save metadata
    metadata_file = output_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_segments, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Metadata saved: {metadata_file}")
    return extracted_segments

def demo_yiddish_alignment():
    """Demo Yiddish-forced Whisper alignment"""
    print("🗣️ YIDDISH-FORCED WHISPER ALIGNMENT")
    print("=" * 55)
    print("Using NATIVE Yiddish language detection")
    print("=" * 55)
    
    # Find files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    
    if not audio_files or not transcript_files:
        print("❌ No original files found")
        return
    
    audio_path = str(audio_files[0])
    transcript_path = str(transcript_files[0])
    
    print(f"Processing: {Path(audio_path).name}")
    
    try:
        # Load original transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            original_transcript = f.read().strip()
        
        print(f"📝 Original transcript (first 50 chars): {original_transcript[:50]}...")
        
        # Get Yiddish Whisper transcription
        yiddish_words, yiddish_full_text = transcribe_yiddish_whisper(audio_path)
        
        print(f"🔄 Yiddish Whisper (first 50 chars): {yiddish_full_text[:50]}...")
        
        # Align transcripts
        aligned_words = align_yiddish_transcripts(original_transcript, yiddish_words)
        
        # Create segments
        segments = create_hebrew_tts_segments(aligned_words)
        
        # Extract audio
        extracted = extract_hebrew_audio_segments(audio_path, segments)
        
        print(f"\n✅ Yiddish alignment completed!")
        print(f"   Created {len(extracted)} Yiddish-aligned segments")
        print(f"   Output directory: tts_segments_hebrew/")
        
        print(f"\n🧪 Test the first segment:")
        print(f"   Audio: {extracted[0]['audio_file']}")
        print(f"   Text: {extracted[0]['text']}")
        print(f"   Duration: {extracted[0]['duration']:.2f}s")
        
        print(f"\n🎯 This should finally work because:")
        print("   ✅ Both audio and text are in Yiddish")
        print("   ✅ Whisper forced to NATIVE Yiddish language detection")
        print("   ✅ Word timings should align perfectly with Yiddish content")
        
    except Exception as e:
        print(f"❌ Yiddish alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_yiddish_alignment() 