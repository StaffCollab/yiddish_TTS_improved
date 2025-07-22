#!/usr/bin/env python3
"""
Silence-Based Audio Segmentation for TTS
Split audio based on natural speech pauses, then map text sequentially
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def detect_speech_segments(audio_path, silence_threshold=0.01, min_silence_duration=0.5, min_segment_duration=2.0):
    """Detect speech segments by finding silence gaps"""
    print(f"🔍 Detecting speech segments in {Path(audio_path).name}...")
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    audio = audio.squeeze()  # Remove channel dimension if mono
    
    print(f"   Audio: {len(audio)/sr:.1f}s, {sr}Hz")
    
    # Calculate RMS energy in small windows
    window_size = int(0.05 * sr)  # 50ms windows
    hop_size = int(0.01 * sr)     # 10ms hop
    
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = torch.sqrt(torch.mean(window ** 2))
        energy.append(rms.item())
    
    energy = np.array(energy)
    
    # Find silence regions (low energy)
    silence_mask = energy < silence_threshold
    
    # Find speech segments between silences
    segments = []
    in_speech = False
    segment_start = 0
    
    silence_samples_needed = int(min_silence_duration / 0.01)  # Convert to samples
    consecutive_silence = 0
    
    for i, is_silent in enumerate(silence_mask):
        time_pos = i * 0.01  # Convert sample to time
        
        if is_silent:
            consecutive_silence += 1
            if consecutive_silence >= silence_samples_needed and in_speech:
                # End of speech segment
                segment_duration = time_pos - segment_start
                if segment_duration >= min_segment_duration:
                    segments.append({
                        'start': segment_start,
                        'end': time_pos,
                        'duration': segment_duration
                    })
                    print(f"   Segment: {segment_start:.1f}s - {time_pos:.1f}s ({segment_duration:.1f}s)")
                in_speech = False
        else:
            consecutive_silence = 0
            if not in_speech:
                # Start of new speech segment
                segment_start = time_pos
                in_speech = True
    
    # Handle last segment if still in speech
    if in_speech:
        final_time = len(energy) * 0.01
        segment_duration = final_time - segment_start
        if segment_duration >= min_segment_duration:
            segments.append({
                'start': segment_start,
                'end': final_time,
                'duration': segment_duration
            })
            print(f"   Final segment: {segment_start:.1f}s - {final_time:.1f}s ({segment_duration:.1f}s)")
    
    print(f"   ✅ Found {len(segments)} speech segments")
    return segments

def map_text_to_segments(transcript_text, audio_segments, max_words_per_segment=12):
    """Map text to audio segments, limiting words per segment for TTS training"""
    print(f"📝 Mapping text to segments (max {max_words_per_segment} words each)...")
    
    words = transcript_text.strip().split()
    print(f"   Total words: {len(words)}")
    
    mapped_segments = []
    word_index = 0
    
    # Create smaller segments by splitting long audio segments if needed
    for i, audio_seg in enumerate(audio_segments):
        audio_duration = audio_seg['duration']
        
        # If audio segment is very long, we'll need to split it into multiple text segments
        if audio_duration > 15:  # Long audio segment
            # Calculate how many sub-segments we need
            num_subsegments = max(1, int(audio_duration / 8))  # Target ~8 seconds each
            words_per_subsegment = min(max_words_per_segment, max(3, max_words_per_segment // 2))
        else:
            num_subsegments = 1
            words_per_subsegment = min(max_words_per_segment, max(3, int(audio_duration * 1.5)))  # ~1.5 words/second
        
        # Create sub-segments from this audio segment
        segment_start_time = audio_seg['start']
        subsegment_duration = audio_duration / num_subsegments
        
        for sub_i in range(num_subsegments):
            if word_index >= len(words):
                break
                
            # Get words for this sub-segment
            segment_words = []
            for j in range(words_per_subsegment):
                if word_index < len(words):
                    segment_words.append(words[word_index])
                    word_index += 1
                else:
                    break
            
            if segment_words:  # Only add if we have words
                # Calculate timing for this sub-segment
                start_time = segment_start_time + (sub_i * subsegment_duration)
                end_time = segment_start_time + ((sub_i + 1) * subsegment_duration)
                
                segment_text = ' '.join(segment_words)
                mapped_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'text': segment_text,
                    'word_count': len(segment_words)
                })
                
                print(f"   Segment {len(mapped_segments)}: {end_time - start_time:.1f}s, {len(segment_words)} words")
                print(f"     Text: {segment_text[:50]}...")
    
    # Handle remaining words if any
    if word_index < len(words):
        remaining_words = words[word_index:]
        # Create additional short segments for remaining words
        while remaining_words:
            chunk_words = remaining_words[:max_words_per_segment]
            remaining_words = remaining_words[max_words_per_segment:]
            
            if mapped_segments:
                # Extend timing from last segment
                last_end = mapped_segments[-1]['end']
                estimated_duration = len(chunk_words) / 2.0  # ~2 words per second
                
                segment_text = ' '.join(chunk_words)
                mapped_segments.append({
                    'start': last_end,
                    'end': last_end + estimated_duration,
                    'duration': estimated_duration,
                    'text': segment_text,
                    'word_count': len(chunk_words)
                })
                
                print(f"   Extra segment {len(mapped_segments)}: {estimated_duration:.1f}s, {len(chunk_words)} words")
                print(f"     Text: {segment_text[:50]}...")
    
    print(f"   ✅ Mapped {len(mapped_segments)} text segments")
    return mapped_segments

def extract_silence_based_segments(audio_path, segments, output_dir="tts_segments_silence"):
    """Extract audio segments based on silence detection"""
    print(f"🎵 Extracting silence-based segments to {output_dir}/")
    
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
            segment_name = f"silence_segment_{i+1:04d}"
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
    
    print(f"   📄 Metadata: {metadata_file}")
    return extracted_segments

def demo_silence_based_alignment():
    """Demo silence-based audio segmentation"""
    print("🔇 SILENCE-BASED AUDIO SEGMENTATION")
    print("=" * 50)
    print("Split audio at natural speech pauses → Map text sequentially")
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
        
        # Detect speech segments based on silence
        audio_segments = detect_speech_segments(audio_path)
        
        if not audio_segments:
            print("❌ No speech segments found!")
            return
        
        # Map text to audio segments
        mapped_segments = map_text_to_segments(original_transcript, audio_segments)
        
        # Extract audio segments
        extracted = extract_silence_based_segments(audio_path, mapped_segments)
        
        print(f"\n✅ SILENCE-BASED SEGMENTATION COMPLETE!")
        print(f"   📁 Output: tts_segments_silence/")
        print(f"   🎵 Audio segments: {len(extracted)}")
        
        if extracted:
            print(f"\n🧪 First segment:")
            print(f"   Text: {extracted[0]['text']}")
            print(f"   Duration: {extracted[0]['duration']:.2f}s")
            print(f"   File: {extracted[0]['audio_file']}")
            
            print(f"\n🎯 This should work better because:")
            print("   ✅ Splits at natural speech pauses")
            print("   ✅ Uses actual audio energy, not Whisper timing")
            print("   ✅ Maps text sequentially to speech segments")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_silence_based_alignment() 