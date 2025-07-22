#!/usr/bin/env python3
"""
Re-segment Original Files for TTS Training
Create proper audio-text aligned segments from original long files
"""

import os
import json
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import re

def analyze_original_file(audio_path, transcript_path):
    """Analyze an original audio-transcript pair"""
    print(f"\n📁 Analyzing: {audio_path}")
    
    # Load audio
    try:
        audio, sr = torchaudio.load(audio_path)
        duration = audio.shape[1] / sr
        print(f"   Audio: {duration:.1f} seconds, {audio.shape[1]} samples")
    except Exception as e:
        print(f"   ❌ Audio error: {e}")
        return None
    
    # Load transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_lines = f.readlines()
        
        # Clean transcript
        transcript_text = ''.join(transcript_lines).strip()
        word_count = len(transcript_text.split())
        char_count = len(transcript_text.replace(' ', ''))
        
        print(f"   Text: {word_count} words, {char_count} characters")
        print(f"   Rate: {word_count/duration*60:.1f} WPM, {char_count/duration:.1f} chars/sec")
        print(f"   First line: {transcript_lines[0][:60]}...")
        
        return {
            'audio_path': audio_path,
            'transcript_path': transcript_path,
            'duration': duration,
            'transcript': transcript_text,
            'lines': transcript_lines,
            'word_count': word_count,
            'char_count': char_count
        }
        
    except Exception as e:
        print(f"   ❌ Transcript error: {e}")
        return None

def detect_sentence_boundaries(text):
    """Detect natural sentence boundaries for segmentation"""
    # Look for sentence endings in Hebrew text
    sentences = []
    current_sentence = ""
    
    # Split by common sentence endings
    # Hebrew punctuation: period, question mark, exclamation
    sentence_endings = ['.', '?', '!', ':', '…']
    
    for char in text:
        current_sentence += char
        if char in sentence_endings:
            # Check if this is end of sentence (not abbreviation)
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    # Add remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    return sentences

def estimate_speech_timing(sentences, total_duration):
    """Estimate timing for each sentence based on character count"""
    total_chars = sum(len(s.replace(' ', '')) for s in sentences)
    timings = []
    current_time = 0
    
    for sentence in sentences:
        sentence_chars = len(sentence.replace(' ', ''))
        sentence_duration = (sentence_chars / total_chars) * total_duration
        
        start_time = current_time
        end_time = current_time + sentence_duration
        
        timings.append({
            'text': sentence,
            'start': start_time,
            'end': end_time,
            'duration': sentence_duration
        })
        
        current_time = end_time
    
    return timings

def create_proper_segments(file_info):
    """Create properly aligned segments from original file"""
    print(f"\n🔄 Creating segments for {Path(file_info['audio_path']).name}")
    
    # Detect sentences
    sentences = detect_sentence_boundaries(file_info['transcript'])
    print(f"   Found {len(sentences)} sentences")
    
    # Estimate timing
    timings = estimate_speech_timing(sentences, file_info['duration'])
    
    # Filter segments by quality
    good_segments = []
    for i, timing in enumerate(timings):
        # Quality checks
        duration = timing['duration']
        word_count = len(timing['text'].split())
        char_count = len(timing['text'].replace(' ', ''))
        
        # Reasonable segment criteria
        if (2.0 <= duration <= 15.0 and  # Duration between 2-15 seconds
            word_count >= 3 and           # At least 3 words
            char_count >= 10 and          # At least 10 characters
            char_count / duration <= 25): # Not too dense
            
            good_segments.append({
                'text': timing['text'],
                'start': timing['start'],
                'end': timing['end'],
                'duration': duration,
                'source_file': file_info['audio_path'],
                'quality_score': min(duration/10, 1.0) * min(word_count/10, 1.0)
            })
    
    print(f"   → {len(good_segments)} quality segments")
    return good_segments

def extract_audio_segment(source_path, start_time, end_time, output_path):
    """Extract audio segment from source file"""
    try:
        # Load full audio
        audio, sr = torchaudio.load(source_path)
        
        # Calculate sample indices
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment
        segment = audio[:, start_sample:end_sample]
        
        # Save segment
        torchaudio.save(output_path, segment, sr)
        return True
        
    except Exception as e:
        print(f"   ❌ Error extracting {output_path}: {e}")
        return False

def resegment_all_files():
    """Re-segment all original files"""
    print("🎯 Re-segmenting Original Files for Proper TTS Training")
    print("=" * 60)
    
    # Find all original files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))
    
    print(f"Found {len(audio_files)} audio files, {len(transcript_files)} transcripts")
    
    if len(audio_files) != len(transcript_files):
        print("❌ Mismatch between audio and transcript files!")
        return
    
    # Analyze all files first
    all_file_info = []
    for audio_file, transcript_file in zip(audio_files, transcript_files):
        info = analyze_original_file(str(audio_file), str(transcript_file))
        if info:
            all_file_info.append(info)
    
    print(f"\n📊 Successfully analyzed {len(all_file_info)} files")
    
    # Create new segments directory
    new_segments_dir = Path("tts_segments_new")
    new_segments_dir.mkdir(exist_ok=True)
    (new_segments_dir / "audio").mkdir(exist_ok=True)
    (new_segments_dir / "text").mkdir(exist_ok=True)
    
    # Process each file
    all_segments = []
    segment_counter = 1
    
    for file_info in all_file_info:
        segments = create_proper_segments(file_info)
        
        for segment in segments:
            # Create segment files
            segment_name = f"segment_{segment_counter:04d}"
            audio_output = new_segments_dir / "audio" / f"{segment_name}.wav"
            text_output = new_segments_dir / "text" / f"{segment_name}.txt"
            
            # Extract audio
            if extract_audio_segment(
                segment['source_file'], 
                segment['start'], 
                segment['end'], 
                str(audio_output)
            ):
                # Save text
                with open(text_output, 'w', encoding='utf-8') as f:
                    f.write(segment['text'])
                
                # Add to metadata
                all_segments.append({
                    'audio_file': f"tts_segments_new/audio/{segment_name}.wav",
                    'text_file': f"tts_segments_new/text/{segment_name}.txt",
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration'],
                    'source_audio': segment['source_file'],
                    'quality_score': segment['quality_score']
                })
                
                segment_counter += 1
    
    # Save new metadata
    metadata_file = new_segments_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Re-segmentation Complete!")
    print(f"   Original segments: 272 (8-second chunks)")
    print(f"   New segments: {len(all_segments)} (natural boundaries)")
    print(f"   Output directory: {new_segments_dir}")
    print(f"   Metadata: {metadata_file}")
    
    # Quality summary
    avg_duration = np.mean([s['duration'] for s in all_segments])
    avg_quality = np.mean([s['quality_score'] for s in all_segments])
    
    print(f"\n📈 Quality Metrics:")
    print(f"   Average duration: {avg_duration:.1f} seconds")
    print(f"   Average quality score: {avg_quality:.2f}")
    print(f"   Duration range: {min(s['duration'] for s in all_segments):.1f} - {max(s['duration'] for s in all_segments):.1f} seconds")
    
    print(f"\n🔧 Next Steps:")
    print(f"1. Review segments in {new_segments_dir}")
    print(f"2. Test alignment: python verify_audio_text_alignment.py")
    print(f"3. Update training script to use new segments")
    print(f"4. Compare TTS training results")

if __name__ == "__main__":
    resegment_all_files() 