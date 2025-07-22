#!/usr/bin/env python3
"""
TTS-Optimal Audio Segmentation
Find natural speech boundaries, then split into TTS-optimal lengths (2-10 seconds)
"""

import os
import json
import torchaudio
import librosa
import numpy as np
from pathlib import Path

def find_detailed_speech_analysis(audio_path):
    """Detailed analysis of speech patterns for TTS segmentation"""
    print(f"🎵 Detailed speech analysis: {Path(audio_path).name}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    # Multiple analysis approaches
    hop_length = 512
    frame_length = 2048
    
    # 1. RMS energy analysis
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms)
    
    # 2. Spectral centroid (pitch-like measure)
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    
    # 3. Zero crossing rate (voice activity)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Find speech activity using multiple criteria
    silence_threshold = np.percentile(rms_db, 25)  # Bottom 25% as silence
    
    # Combine criteria for better speech detection
    is_speech = (
        (rms_db > silence_threshold) &  # Has energy
        (spec_centroid > 500) &         # Has vocal-like frequency content
        (zcr < 0.1)                     # Not too noisy
    )
    
    print(f"   Duration: {duration:.1f}s")
    print(f"   Speech activity: {np.mean(is_speech)*100:.1f}%")
    
    return {
        'audio': audio,
        'sr': sr,
        'times': times,
        'is_speech': is_speech,
        'rms_db': rms_db,
        'duration': duration
    }

def find_natural_break_points(speech_analysis, min_pause_duration=0.3):
    """Find natural break points (pauses) in speech"""
    is_speech = speech_analysis['is_speech']
    times = speech_analysis['times']
    
    # Find transitions from speech to silence
    speech_changes = np.diff(is_speech.astype(int))
    
    # Find pause boundaries
    pause_starts = times[1:][speech_changes == -1]  # Speech -> Silence
    pause_ends = times[1:][speech_changes == 1]     # Silence -> Speech
    
    # Filter pauses by minimum duration
    natural_breaks = []
    
    for i, start in enumerate(pause_starts):
        if i < len(pause_ends):
            pause_duration = pause_ends[i] - start
            if pause_duration >= min_pause_duration:
                natural_breaks.append({
                    'time': pause_ends[i],  # End of pause = good break point
                    'pause_duration': pause_duration,
                    'confidence': min(pause_duration / 1.0, 1.0)  # Max confidence at 1s pause
                })
    
    print(f"   Found {len(natural_breaks)} natural break points")
    return natural_breaks

def create_tts_optimal_segments(speech_analysis, natural_breaks, transcript_text, 
                               target_duration_range=(3, 8), max_duration=12):
    """Create TTS-optimal segments using natural breaks"""
    print(f"🎯 Creating TTS-optimal segments (target: {target_duration_range[0]}-{target_duration_range[1]}s)")
    
    total_duration = speech_analysis['duration']
    
    # Add start and end points to break points
    all_breaks = [{'time': 0, 'confidence': 1.0}]  # Start
    all_breaks.extend(natural_breaks)
    all_breaks.append({'time': total_duration, 'confidence': 1.0})  # End
    
    # Sort by time
    all_breaks.sort(key=lambda x: x['time'])
    break_times = [b['time'] for b in all_breaks]
    
    print(f"   Working with {len(break_times)} potential break points")
    
    # Create segments with optimal durations
    segments = []
    current_start = 0
    
    i = 1  # Start from first real break (skip start point)
    while i < len(break_times) and current_start < total_duration:
        current_end = break_times[i]
        segment_duration = current_end - current_start
        
        # Check if this segment is good for TTS
        if target_duration_range[0] <= segment_duration <= target_duration_range[1]:
            # Perfect duration - use this segment
            segments.append({
                'start': current_start,
                'end': current_end,
                'duration': segment_duration,
                'quality': 'optimal'
            })
            current_start = current_end
            i += 1
            
        elif segment_duration < target_duration_range[0]:
            # Too short - try to extend to next break
            if i + 1 < len(break_times):
                next_end = break_times[i + 1]
                extended_duration = next_end - current_start
                
                if extended_duration <= max_duration:
                    # Extend to next break
                    i += 1
                    continue
                else:
                    # Even extension is too long - use current segment anyway
                    segments.append({
                        'start': current_start,
                        'end': current_end,
                        'duration': segment_duration,
                        'quality': 'short'
                    })
                    current_start = current_end
                    i += 1
            else:
                # Last segment - use what we have
                segments.append({
                    'start': current_start,
                    'end': current_end,
                    'duration': segment_duration,
                    'quality': 'short'
                })
                break
                
        else:
            # Too long - need to split
            # Try to find a break point within target range
            target_end = current_start + target_duration_range[1]
            
            # Find closest break point to target
            best_break_idx = i
            best_break_time = break_times[i]
            
            for j in range(i, len(break_times)):
                break_time = break_times[j]
                if break_time > target_end:
                    break
                if break_time - current_start >= target_duration_range[0]:
                    best_break_idx = j
                    best_break_time = break_time
            
            # Use the best break point found
            actual_duration = best_break_time - current_start
            segments.append({
                'start': current_start,
                'end': best_break_time,
                'duration': actual_duration,
                'quality': 'good' if actual_duration <= target_duration_range[1] else 'long'
            })
            
            current_start = best_break_time
            # Find where we are in the break list
            i = best_break_idx + 1
    
    print(f"   Created {len(segments)} TTS-optimal segments")
    
    # Quality summary
    quality_counts = {}
    for seg in segments:
        quality_counts[seg['quality']] = quality_counts.get(seg['quality'], 0) + 1
    
    print(f"   Quality distribution: {quality_counts}")
    
    return segments

def map_text_to_tts_segments(segments, transcript_text):
    """Map transcript text to TTS-optimal segments"""
    print(f"📝 Mapping text to {len(segments)} TTS segments")
    
    # Calculate total duration for proportional text distribution
    total_duration = sum(seg['duration'] for seg in segments)
    total_chars = len(transcript_text.replace(' ', ''))
    
    mappings = []
    text_position = 0
    
    for i, segment in enumerate(segments):
        # Calculate proportional text amount
        segment_ratio = segment['duration'] / total_duration
        target_chars = int(total_chars * segment_ratio)
        
        # Get remaining text
        remaining_text = transcript_text[text_position:]
        
        if i == len(segments) - 1:
            # Last segment gets all remaining text
            segment_text = remaining_text.strip()
        else:
            # Find good break point near target
            if target_chars >= len(remaining_text):
                segment_text = remaining_text.strip()
            else:
                # Look for sentence or phrase boundary
                search_window = 100  # Characters to search around target
                search_start = max(0, target_chars - search_window)
                search_end = min(len(remaining_text), target_chars + search_window)
                
                # Prefer sentence endings, then periods, then spaces
                best_break = target_chars
                for punct in ['.', '!', '?', ':', '…']:
                    for pos in range(search_start, search_end):
                        if pos < len(remaining_text) and remaining_text[pos] == punct:
                            if abs(pos - target_chars) < abs(best_break - target_chars):
                                best_break = pos + 1  # Include punctuation
                
                # If no punctuation found, look for spaces
                if best_break == target_chars:
                    for pos in range(search_start, search_end):
                        if pos < len(remaining_text) and remaining_text[pos] == ' ':
                            if abs(pos - target_chars) < abs(best_break - target_chars):
                                best_break = pos
                
                segment_text = remaining_text[:best_break].strip()
        
        if segment_text:
            # Calculate speech rate for quality check
            word_count = len(segment_text.split())
            char_count = len(segment_text.replace(' ', ''))
            wpm = (word_count / segment['duration']) * 60
            cps = char_count / segment['duration']
            
            mappings.append({
                'audio_segment': segment,
                'text': segment_text,
                'word_count': word_count,
                'char_count': char_count,
                'wpm': wpm,
                'cps': cps,
                'tts_quality': 'good' if 100 <= wpm <= 250 and 8 <= cps <= 20 else 'questionable'
            })
            
            text_position += len(segment_text)
            # Skip any whitespace
            while text_position < len(transcript_text) and transcript_text[text_position] == ' ':
                text_position += 1
    
    return mappings

def analyze_tts_suitability(mappings):
    """Analyze how suitable the segments are for TTS training"""
    print("📊 TTS Suitability Analysis")
    
    total_segments = len(mappings)
    good_duration = sum(1 for m in mappings if 2 <= m['audio_segment']['duration'] <= 10)
    good_speech_rate = sum(1 for m in mappings if m['tts_quality'] == 'good')
    
    duration_suitability = good_duration / total_segments if total_segments > 0 else 0
    speech_rate_suitability = good_speech_rate / total_segments if total_segments > 0 else 0
    overall_suitability = (duration_suitability + speech_rate_suitability) / 2
    
    print(f"   Duration suitability: {duration_suitability*100:.1f}% ({good_duration}/{total_segments})")
    print(f"   Speech rate suitability: {speech_rate_suitability*100:.1f}% ({good_speech_rate}/{total_segments})")
    print(f"   Overall TTS suitability: {overall_suitability*100:.1f}%")
    
    # Show problematic segments
    problems = [m for m in mappings if m['tts_quality'] == 'questionable' or 
                m['audio_segment']['duration'] < 2 or m['audio_segment']['duration'] > 10]
    
    if problems:
        print(f"   Problematic segments: {len(problems)}")
        for i, p in enumerate(problems[:3]):  # Show first 3
            dur = p['audio_segment']['duration']
            wpm = p['wpm']
            text_preview = p['text'][:40] + "..." if len(p['text']) > 40 else p['text']
            print(f"     {i+1}: {dur:.1f}s, {wpm:.0f}WPM - {text_preview}")
    
    return overall_suitability

def preview_tts_segmentation():
    """Preview TTS-optimal segmentation on first few files"""
    print("🎯 TTS-Optimal Segmentation Preview")
    print("=" * 50)
    
    # Test on first 2 files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:2]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:2]
    
    overall_suitability = []
    
    for audio_file, transcript_file in zip(audio_files, transcript_files):
        print(f"\n📁 Processing: {audio_file.name}")
        
        # Load transcript
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read().strip()
        
        # Analyze audio
        speech_analysis = find_detailed_speech_analysis(str(audio_file))
        
        # Find break points
        natural_breaks = find_natural_break_points(speech_analysis)
        
        # Create TTS-optimal segments
        segments = create_tts_optimal_segments(speech_analysis, natural_breaks, transcript_text)
        
        # Map text to segments
        mappings = map_text_to_tts_segments(segments, transcript_text)
        
        # Analyze suitability
        suitability = analyze_tts_suitability(mappings)
        overall_suitability.append(suitability)
        
        # Show sample mappings
        print(f"\n📋 Sample segments from {audio_file.name}:")
        for i, mapping in enumerate(mappings[:3]):
            seg = mapping['audio_segment']
            text = mapping['text'][:50] + "..." if len(mapping['text']) > 50 else mapping['text']
            print(f"   {i+1}: {seg['start']:.1f}-{seg['end']:.1f}s ({seg['duration']:.1f}s)")
            print(f"      Rate: {mapping['wpm']:.0f}WPM, Quality: {mapping['tts_quality']}")
            print(f"      Text: {text}")
    
    avg_suitability = np.mean(overall_suitability) if overall_suitability else 0
    print(f"\n🎯 Overall TTS Suitability: {avg_suitability*100:.1f}%")
    
    if avg_suitability > 0.75:
        print("✅ Excellent - segments are well-suited for TTS training")
    elif avg_suitability > 0.6:
        print("✅ Good - segments should work well for TTS training")
    elif avg_suitability > 0.4:
        print("⚠️  Fair - TTS training possible but may need refinement")
    else:
        print("❌ Poor - segments need significant improvement for TTS")
    
    print(f"\n🔧 Next Steps:")
    if avg_suitability > 0.6:
        print("1. Proceed with full TTS-optimal segmentation")
        print("2. Use these segments for conservative training")
    else:
        print("1. Adjust segmentation parameters")
        print("2. Consider manual segmentation for problem files")
    print("3. Test training with a few segments first")

if __name__ == "__main__":
    preview_tts_segmentation() 