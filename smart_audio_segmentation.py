#!/usr/bin/env python3
"""
Smart Audio Segmentation for TTS
Analyze actual audio to find speech boundaries, then map text appropriately
"""

import os
import json
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def find_speech_boundaries(audio_path, min_silence_duration=0.5, silence_threshold_db=-40):
    """Find actual speech/silence boundaries in audio"""
    print(f"🎵 Analyzing speech boundaries: {Path(audio_path).name}")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    
    # Convert to dB
    audio_db = librosa.amplitude_to_db(np.abs(audio))
    
    # Find frames below silence threshold
    hop_length = 512
    frame_length = 2048
    
    # RMS energy per frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms)
    
    # Times for each frame
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Find silence regions
    silence_frames = rms_db < silence_threshold_db
    
    # Find speech segments (consecutive non-silence)
    speech_segments = []
    in_speech = False
    segment_start = 0
    
    for i, is_silent in enumerate(silence_frames):
        if not is_silent and not in_speech:
            # Start of speech
            segment_start = times[i]
            in_speech = True
        elif is_silent and in_speech:
            # End of speech
            segment_end = times[i]
            if segment_end - segment_start >= 1.0:  # At least 1 second
                speech_segments.append({
                    'start': segment_start,
                    'end': segment_end,
                    'duration': segment_end - segment_start
                })
            in_speech = False
    
    # Handle case where audio ends during speech
    if in_speech:
        speech_segments.append({
            'start': segment_start,
            'end': times[-1],
            'duration': times[-1] - segment_start
        })
    
    print(f"   Found {len(speech_segments)} speech segments")
    for i, seg in enumerate(speech_segments[:5]):  # Show first 5
        print(f"   {i+1}: {seg['start']:.1f}s - {seg['end']:.1f}s ({seg['duration']:.1f}s)")
    
    return speech_segments, audio, sr

def analyze_original_transcript(transcript_path):
    """Analyze original transcript structure"""
    print(f"📄 Analyzing transcript: {Path(transcript_path).name}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and analyze
    clean_lines = [line.strip() for line in lines if line.strip()]
    total_text = ' '.join(clean_lines)
    
    print(f"   Lines: {len(clean_lines)}")
    print(f"   Total characters: {len(total_text)}")
    print(f"   Total words: {len(total_text.split())}")
    
    # Try to detect natural break points
    # Look for lines that end with sentence-ending punctuation
    break_points = []
    for i, line in enumerate(clean_lines):
        if line.endswith(('.', '!', '?', ':', '…')):
            break_points.append(i)
    
    print(f"   Natural break points: {len(break_points)}")
    
    return {
        'lines': clean_lines,
        'total_text': total_text,
        'break_points': break_points
    }

def map_text_to_audio_segments(speech_segments, transcript_info):
    """Map text content to actual audio speech segments"""
    print(f"🔗 Mapping text to {len(speech_segments)} audio segments")
    
    lines = transcript_info['lines']
    total_chars = len(transcript_info['total_text'].replace(' ', ''))
    
    # Strategy 1: If we have similar number of lines and speech segments
    if abs(len(lines) - len(speech_segments)) <= 2:
        print("   Using line-to-segment mapping")
        mappings = []
        for i, segment in enumerate(speech_segments):
            if i < len(lines):
                mappings.append({
                    'audio_segment': segment,
                    'text': lines[i],
                    'mapping_confidence': 'high'
                })
        return mappings
    
    # Strategy 2: Distribute text proportionally based on segment duration
    print("   Using proportional duration mapping")
    total_duration = sum(seg['duration'] for seg in speech_segments)
    
    mappings = []
    text_position = 0
    
    for segment in speech_segments:
        # Calculate how much text should go in this segment
        segment_ratio = segment['duration'] / total_duration
        chars_for_segment = int(total_chars * segment_ratio)
        
        # Find text boundaries (try to break at word boundaries)
        remaining_text = transcript_info['total_text'][text_position:]
        
        if chars_for_segment >= len(remaining_text):
            # Last segment gets all remaining text
            segment_text = remaining_text
        else:
            # Find word boundary near the target character count
            target_end = min(chars_for_segment, len(remaining_text))
            
            # Look for space near target position
            search_start = max(0, target_end - 50)
            search_end = min(len(remaining_text), target_end + 50)
            
            best_break = target_end
            for i in range(search_start, search_end):
                if remaining_text[i] == ' ':
                    if abs(i - target_end) < abs(best_break - target_end):
                        best_break = i
            
            segment_text = remaining_text[:best_break].strip()
        
        if segment_text:
            mappings.append({
                'audio_segment': segment,
                'text': segment_text,
                'mapping_confidence': 'medium'
            })
            text_position += len(segment_text) + 1  # +1 for space
    
    print(f"   Created {len(mappings)} text-audio mappings")
    return mappings

def validate_mapping_quality(mappings):
    """Validate the quality of text-audio mappings"""
    print("✅ Validating mapping quality")
    
    quality_issues = []
    
    for i, mapping in enumerate(mappings):
        duration = mapping['audio_segment']['duration']
        text = mapping['text']
        word_count = len(text.split())
        char_count = len(text.replace(' ', ''))
        
        # Calculate speech rate
        wpm = (word_count / duration) * 60 if duration > 0 else 0
        cps = char_count / duration if duration > 0 else 0
        
        # Quality checks
        issues = []
        if wpm > 300 or wpm < 60:
            issues.append(f"unusual WPM: {wpm:.1f}")
        if cps > 25 or cps < 5:
            issues.append(f"unusual chars/sec: {cps:.1f}")
        if duration < 1.0:
            issues.append(f"very short: {duration:.1f}s")
        if duration > 20.0:
            issues.append(f"very long: {duration:.1f}s")
        
        if issues:
            quality_issues.append({
                'segment': i + 1,
                'issues': issues,
                'text_preview': text[:40] + "...",
                'duration': duration
            })
    
    print(f"   Quality issues in {len(quality_issues)}/{len(mappings)} segments")
    
    if quality_issues:
        print("   Issues found:")
        for issue in quality_issues[:5]:  # Show first 5
            print(f"   Segment {issue['segment']}: {', '.join(issue['issues'])}")
            print(f"     Text: {issue['text_preview']}")
    
    return len(quality_issues) / len(mappings) if mappings else 1.0

def smart_segment_file(audio_path, transcript_path):
    """Smart segmentation of a single audio-transcript pair"""
    print(f"\n🧠 Smart segmentation: {Path(audio_path).name}")
    
    # Step 1: Find actual speech boundaries in audio
    speech_segments, audio, sr = find_speech_boundaries(audio_path)
    
    if len(speech_segments) == 0:
        print("   ❌ No speech segments found")
        return []
    
    # Step 2: Analyze transcript structure
    transcript_info = analyze_original_transcript(transcript_path)
    
    # Step 3: Map text to audio segments
    mappings = map_text_to_audio_segments(speech_segments, transcript_info)
    
    # Step 4: Validate quality
    error_rate = validate_mapping_quality(mappings)
    
    print(f"   Mapping quality: {(1-error_rate)*100:.1f}% good")
    
    return mappings, error_rate

def preview_segmentation():
    """Preview the smart segmentation approach on a few files"""
    print("🔍 Smart Audio Segmentation Preview")
    print("=" * 50)
    
    # Test on first 3 files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:3]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:3]
    
    overall_quality = []
    
    for audio_file, transcript_file in zip(audio_files, transcript_files):
        mappings, error_rate = smart_segment_file(str(audio_file), str(transcript_file))
        overall_quality.append(1 - error_rate)
        
        # Show preview of mappings
        print(f"\n📋 Preview mappings for {audio_file.name}:")
        for i, mapping in enumerate(mappings[:3]):  # First 3 segments
            segment = mapping['audio_segment']
            text = mapping['text'][:50] + "..." if len(mapping['text']) > 50 else mapping['text']
            print(f"   {i+1}: {segment['start']:.1f}-{segment['end']:.1f}s: {text}")
    
    avg_quality = np.mean(overall_quality) if overall_quality else 0
    print(f"\n📊 Overall Quality: {avg_quality*100:.1f}%")
    
    if avg_quality > 0.7:
        print("✅ Smart segmentation looks good - proceed with full processing")
    else:
        print("⚠️  Smart segmentation needs refinement")
    
    print(f"\n🎯 Next Steps:")
    print("1. Review preview results above")
    print("2. If quality is good, run full segmentation")
    print("3. If not, adjust silence detection parameters")
    print("4. Listen to a few segments to verify alignment")

if __name__ == "__main__":
    preview_segmentation() 