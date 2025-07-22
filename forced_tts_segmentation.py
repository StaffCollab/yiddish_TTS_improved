#!/usr/bin/env python3
"""
Forced TTS Segmentation
Create TTS-appropriate segments even in continuous speech by finding best break points
"""

import os
import json
import torchaudio
import librosa
import numpy as np
from pathlib import Path

def find_optimal_break_points(audio_path, target_segment_length=6, max_segment_length=10):
    """Find optimal break points for TTS, even in continuous speech"""
    print(f"✂️  Finding break points: {Path(audio_path).name}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    # Audio analysis for finding best break points
    hop_length = 512
    frame_length = 2048
    
    # RMS energy per frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms)
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    print(f"   Duration: {duration:.1f}s")
    print(f"   Target segments: ~{duration/target_segment_length:.0f} segments of {target_segment_length}s each")
    
    # Strategy: Force segments at intervals, but find best break points within windows
    break_points = [0]  # Start
    current_position = 0
    
    while current_position + max_segment_length < duration:
        # Define search window for next break point
        min_next_break = current_position + target_segment_length - 2  # Allow 2s flexibility
        max_next_break = current_position + max_segment_length
        
        # Find the best break point in this window (lowest energy = quieter moment)
        # Convert time to frame indices
        min_frame = int(min_next_break * sr / hop_length)
        max_frame = int(max_next_break * sr / hop_length)
        
        # Ensure we don't go beyond available frames
        min_frame = max(0, min(min_frame, len(rms) - 1))
        max_frame = max(min_frame + 1, min(max_frame, len(rms)))
        
        if min_frame < max_frame:
            # Find the frame with minimum energy (quietest moment) in the window
            window_energies = rms_db[min_frame:max_frame]
            best_relative_frame = np.argmin(window_energies)
            best_frame = min_frame + best_relative_frame
            best_time = times[best_frame]
        else:
            # Fallback to target time if we can't find a good window
            best_time = current_position + target_segment_length
        
        break_points.append(best_time)
        current_position = best_time
        
        print(f"   Break point at {best_time:.1f}s")
    
    # Add final point
    break_points.append(duration)
    
    print(f"   Created {len(break_points)-1} segments")
    
    return break_points, audio, sr

def create_forced_segments(break_points, audio, sr):
    """Create audio segments from break points"""
    segments = []
    
    for i in range(len(break_points) - 1):
        start_time = break_points[i]
        end_time = break_points[i + 1]
        duration = end_time - start_time
        
        segments.append({
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'segment_index': i + 1
        })
    
    return segments

def distribute_text_to_segments(segments, transcript_text):
    """Distribute transcript text to segments proportionally"""
    print(f"📝 Distributing text to {len(segments)} segments")
    
    total_duration = sum(seg['duration'] for seg in segments)
    total_chars = len(transcript_text.replace(' ', ''))
    
    mappings = []
    text_position = 0
    
    for i, segment in enumerate(segments):
        # Calculate proportional text
        if i == len(segments) - 1:
            # Last segment gets all remaining text
            segment_text = transcript_text[text_position:].strip()
        else:
            # Proportional distribution
            segment_ratio = segment['duration'] / total_duration
            target_chars = int(total_chars * segment_ratio)
            
            # Find good break point (prefer sentence/word boundaries)
            remaining_text = transcript_text[text_position:]
            
            if target_chars >= len(remaining_text):
                segment_text = remaining_text.strip()
            else:
                # Look for good break points near target
                search_radius = min(100, target_chars // 2)
                search_start = max(0, target_chars - search_radius)
                search_end = min(len(remaining_text), target_chars + search_radius)
                
                # Priority: sentence end > phrase end > word boundary
                best_break = target_chars
                
                # Look for sentence endings
                for punct in ['.', '!', '?', ':', '…']:
                    for pos in range(search_start, search_end):
                        if pos < len(remaining_text) and remaining_text[pos] == punct:
                            if abs(pos - target_chars) < abs(best_break - target_chars):
                                best_break = pos + 1  # Include punctuation
                
                # If no sentence end, look for commas or spaces
                if best_break == target_chars:
                    for punct in [',', ' ']:
                        for pos in range(search_start, search_end):
                            if pos < len(remaining_text) and remaining_text[pos] == punct:
                                if abs(pos - target_chars) < abs(best_break - target_chars):
                                    best_break = pos if punct == ' ' else pos + 1
                
                segment_text = remaining_text[:best_break].strip()
        
        if segment_text:
            # Calculate quality metrics
            word_count = len(segment_text.split())
            char_count = len(segment_text.replace(' ', ''))
            wpm = (word_count / segment['duration']) * 60 if segment['duration'] > 0 else 0
            cps = char_count / segment['duration'] if segment['duration'] > 0 else 0
            
            # Quality assessment
            duration_ok = 2 <= segment['duration'] <= 12
            rate_ok = 80 <= wpm <= 300 and 5 <= cps <= 25
            
            mappings.append({
                'segment': segment,
                'text': segment_text,
                'word_count': word_count,
                'char_count': char_count,
                'wpm': wpm,
                'cps': cps,
                'duration_ok': duration_ok,
                'rate_ok': rate_ok,
                'overall_quality': 'good' if duration_ok and rate_ok else 'acceptable'
            })
            
            text_position += len(segment_text)
            # Skip whitespace
            while text_position < len(transcript_text) and transcript_text[text_position] in ' \n\t':
                text_position += 1
    
    return mappings

def extract_audio_segments(audio_path, mappings, output_dir):
    """Extract actual audio segments to files"""
    print(f"🎵 Extracting {len(mappings)} audio segments")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load original audio
    audio, sr = torchaudio.load(audio_path)
    
    extracted_segments = []
    
    for i, mapping in enumerate(mappings):
        segment = mapping['segment']
        
        # Extract audio segment
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio.shape[1] and end_sample <= audio.shape[1]:
            audio_segment = audio[:, start_sample:end_sample]
            
            # Save files
            segment_name = f"segment_{i+1:04d}"
            audio_file = output_dir / "audio" / f"{segment_name}.wav"
            text_file = output_dir / "text" / f"{segment_name}.txt"
            
            torchaudio.save(str(audio_file), audio_segment, sr)
            
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(mapping['text'])
            
            extracted_segments.append({
                'audio_file': str(audio_file),
                'text_file': str(text_file),
                'text': mapping['text'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'wpm': mapping['wpm'],
                'cps': mapping['cps'],
                'quality': mapping['overall_quality']
            })
    
    return extracted_segments

def process_forced_segmentation(audio_path, transcript_path, output_dir="tts_segments_forced"):
    """Complete forced segmentation process"""
    print(f"\n🚀 Processing: {Path(audio_path).name}")
    
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read().strip()
    
    print(f"   Transcript: {len(transcript_text)} characters, {len(transcript_text.split())} words")
    
    # Find break points
    break_points, audio, sr = find_optimal_break_points(audio_path)
    
    # Create segments
    segments = create_forced_segments(break_points, audio, sr)
    
    # Distribute text
    mappings = distribute_text_to_segments(segments, transcript_text)
    
    # Extract audio files
    file_output_dir = Path(output_dir) / Path(audio_path).stem
    extracted = extract_audio_segments(audio_path, mappings, file_output_dir)
    
    # Quality analysis
    good_duration = sum(1 for e in extracted if 2 <= e['duration'] <= 10)
    good_rate = sum(1 for e in extracted if e['quality'] == 'good')
    
    print(f"   Results: {len(extracted)} segments")
    print(f"   Duration quality: {good_duration}/{len(extracted)} segments (2-10s)")
    print(f"   Rate quality: {good_rate}/{len(extracted)} segments")
    print(f"   Output: {file_output_dir}")
    
    return extracted

def demo_forced_segmentation():
    """Demo forced segmentation on first file"""
    print("🚀 Forced TTS Segmentation Demo")
    print("=" * 50)
    
    # Test on first file
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    
    if not audio_files or not transcript_files:
        print("❌ No original files found")
        return
    
    # Process first file
    extracted = process_forced_segmentation(str(audio_files[0]), str(transcript_files[0]))
    
    # Show sample results
    print(f"\n📋 Sample segments:")
    for i, segment in enumerate(extracted[:5]):
        print(f"   {i+1}: {segment['duration']:.1f}s, {segment['wpm']:.0f}WPM")
        print(f"      Quality: {segment['quality']}")
        print(f"      Text: {segment['text'][:50]}...")
        print(f"      File: {Path(segment['audio_file']).name}")
    
    # Overall assessment
    avg_duration = np.mean([s['duration'] for s in extracted])
    tts_suitable = sum(1 for s in extracted if 2 <= s['duration'] <= 10)
    
    print(f"\n📊 Summary:")
    print(f"   Total segments: {len(extracted)}")
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   TTS-suitable: {tts_suitable}/{len(extracted)} ({tts_suitable/len(extracted)*100:.0f}%)")
    
    if tts_suitable / len(extracted) > 0.8:
        print("✅ Excellent - ready for TTS training!")
    elif tts_suitable / len(extracted) > 0.6:
        print("✅ Good - should work well for TTS training")
    else:
        print("⚠️  Needs adjustment - consider changing target segment length")
    
    print(f"\n🔧 Next Steps:")
    print("1. Review generated segments")
    print("2. If quality is good, process all files")
    print("3. Use these segments for TTS training")

if __name__ == "__main__":
    demo_forced_segmentation() 