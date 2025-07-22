#!/usr/bin/env python3
"""
Audio-Text Alignment Verification
Use speech recognition to verify that audio segments match their text
"""

import os
import json
import librosa
import numpy as np
from pathlib import Path
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

def load_whisper_model():
    """Load Whisper model for speech recognition verification"""
    print("🤖 Loading Whisper model for verification...")
    
    try:
        # Use a smaller Whisper model for efficiency
        model_name = "openai/whisper-base"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        print(f"   ✅ Loaded {model_name}")
        return processor, model
    except Exception as e:
        print(f"   ❌ Could not load Whisper: {e}")
        print("   Install with: pip install transformers torch torchaudio")
        return None, None

def transcribe_audio_segment(audio_path, start_time, end_time, processor, model):
    """Transcribe a specific segment of audio"""
    try:
        # Load audio segment
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        
        # Extract segment
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio[start_sample:end_sample]
        
        # Transcribe with Whisper
        inputs = processor(audio_segment, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            predicted_ids = model.generate(inputs["input_features"])
        
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
        
    except Exception as e:
        print(f"   ❌ Transcription error: {e}")
        return ""

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two texts (basic approach)"""
    if not text1 or not text2:
        return 0.0
    
    # Simple character-level similarity
    # Remove spaces and convert to lowercase for comparison
    clean1 = ''.join(text1.lower().split())
    clean2 = ''.join(text2.lower().split())
    
    if len(clean1) == 0 and len(clean2) == 0:
        return 1.0
    if len(clean1) == 0 or len(clean2) == 0:
        return 0.0
    
    # Calculate character overlap
    common_chars = 0
    total_chars = max(len(clean1), len(clean2))
    
    for i, char in enumerate(clean1):
        if i < len(clean2) and char == clean2[i]:
            common_chars += 1
    
    similarity = common_chars / total_chars
    return similarity

def verify_segment_alignment(audio_path, text_segments, processor, model, min_similarity=0.3):
    """Verify that text segments align with audio content"""
    print(f"🔍 Verifying alignment for {Path(audio_path).name}")
    
    alignments = []
    
    for i, segment in enumerate(text_segments):
        print(f"   Segment {i+1}/{len(text_segments)}: {segment['start']:.1f}-{segment['end']:.1f}s")
        
        # Transcribe the audio segment
        transcribed = transcribe_audio_segment(
            audio_path, segment['start'], segment['end'], processor, model
        )
        
        # Compare with expected text
        expected_text = segment['text']
        similarity = calculate_text_similarity(transcribed, expected_text)
        
        alignment_result = {
            'segment_index': i + 1,
            'start': segment['start'],
            'end': segment['end'],
            'duration': segment['end'] - segment['start'],
            'expected_text': expected_text,
            'transcribed_text': transcribed,
            'similarity': similarity,
            'aligned': similarity >= min_similarity,
            'status': 'good' if similarity >= min_similarity else 'poor'
        }
        
        alignments.append(alignment_result)
        
        # Show immediate feedback
        status_emoji = "✅" if alignment_result['aligned'] else "❌"
        print(f"      {status_emoji} Similarity: {similarity:.2f}")
        print(f"      Expected: {expected_text[:40]}...")
        print(f"      Heard:    {transcribed[:40]}...")
    
    return alignments

def analyze_alignment_quality(alignments):
    """Analyze overall alignment quality"""
    print("\n📊 Alignment Quality Analysis")
    print("=" * 40)
    
    total_segments = len(alignments)
    good_alignments = sum(1 for a in alignments if a['aligned'])
    avg_similarity = np.mean([a['similarity'] for a in alignments])
    
    print(f"Total segments: {total_segments}")
    print(f"Good alignments: {good_alignments}/{total_segments} ({good_alignments/total_segments*100:.1f}%)")
    print(f"Average similarity: {avg_similarity:.2f}")
    
    # Show problematic segments
    poor_segments = [a for a in alignments if not a['aligned']]
    if poor_segments:
        print(f"\n❌ Problematic segments ({len(poor_segments)}):")
        for seg in poor_segments[:5]:  # Show first 5
            print(f"   Segment {seg['segment_index']}: {seg['similarity']:.2f} similarity")
            print(f"      Expected: {seg['expected_text'][:50]}...")
            print(f"      Heard:    {seg['transcribed_text'][:50]}...")
    
    # Overall assessment
    if good_alignments / total_segments > 0.8:
        print("\n✅ Excellent alignment - ready for TTS training")
        recommendation = "proceed"
    elif good_alignments / total_segments > 0.6:
        print("\n⚠️  Fair alignment - some segments may need manual review")
        recommendation = "review"
    else:
        print("\n❌ Poor alignment - segmentation needs major improvement")
        recommendation = "redo"
    
    return {
        'total_segments': total_segments,
        'good_alignments': good_alignments,
        'alignment_rate': good_alignments / total_segments,
        'avg_similarity': avg_similarity,
        'recommendation': recommendation,
        'poor_segments': poor_segments
    }

def create_test_segments_from_forced():
    """Create test segments using forced segmentation for verification"""
    print("🧪 Creating test segments for alignment verification")
    
    # Use the forced segmentation approach but only create a few test segments
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    
    if not audio_files or not transcript_files:
        print("❌ No original files found")
        return []
    
    audio_path = str(audio_files[0])
    transcript_path = str(transcript_files[0])
    
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read().strip()
    
    # Load audio to get duration
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    print(f"   Audio: {Path(audio_path).name} ({duration:.1f}s)")
    print(f"   Text: {len(transcript_text)} characters")
    
    # Create a few test segments (first 30 seconds, 6-second segments)
    test_segments = []
    segment_length = 6
    max_test_duration = 30  # Only test first 30 seconds
    
    total_chars = len(transcript_text.replace(' ', ''))
    
    for i in range(int(max_test_duration / segment_length)):
        start_time = i * segment_length
        end_time = min(start_time + segment_length, duration, max_test_duration)
        
        if end_time <= start_time:
            break
        
        # Calculate proportional text for this segment
        time_ratio = (end_time - start_time) / min(duration, max_test_duration)
        start_char_ratio = start_time / min(duration, max_test_duration)
        
        start_char = int(start_char_ratio * total_chars)
        segment_chars = int(time_ratio * total_chars)
        
        # Find text segment with word boundaries
        text_without_spaces = transcript_text.replace(' ', '')
        char_count = 0
        text_start = 0
        
        # Find start position
        for pos, char in enumerate(transcript_text):
            if char != ' ':
                if char_count == start_char:
                    text_start = pos
                    break
                char_count += 1
        
        # Find end position
        chars_found = 0
        text_end = len(transcript_text)
        for pos in range(text_start, len(transcript_text)):
            if transcript_text[pos] != ' ':
                chars_found += 1
                if chars_found >= segment_chars:
                    # Look for word boundary
                    while pos < len(transcript_text) and transcript_text[pos] not in ' .,!?':
                        pos += 1
                    text_end = pos
                    break
        
        segment_text = transcript_text[text_start:text_end].strip()
        
        if segment_text:
            test_segments.append({
                'start': start_time,
                'end': end_time,
                'text': segment_text
            })
    
    print(f"   Created {len(test_segments)} test segments for verification")
    return test_segments, audio_path

def demo_alignment_verification():
    """Demo the alignment verification process"""
    print("🔍 Audio-Text Alignment Verification Demo")
    print("=" * 50)
    
    # Load speech recognition model
    processor, model = load_whisper_model()
    if not processor or not model:
        print("❌ Cannot verify alignment without speech recognition model")
        print("   Install required packages:")
        print("   pip install transformers torch torchaudio")
        return
    
    # Create test segments
    try:
        test_segments, audio_path = create_test_segments_from_forced()
        if not test_segments:
            return
        
        # Verify alignment
        alignments = verify_segment_alignment(audio_path, test_segments, processor, model)
        
        # Analyze results
        quality = analyze_alignment_quality(alignments)
        
        print(f"\n🎯 Conclusion:")
        if quality['recommendation'] == 'proceed':
            print("✅ Forced segmentation produces good alignment")
            print("   → Safe to proceed with full dataset segmentation")
        elif quality['recommendation'] == 'review':
            print("⚠️  Forced segmentation has some issues")
            print("   → Consider manual review or different approach")
        else:
            print("❌ Forced segmentation produces poor alignment")
            print("   → Need better segmentation strategy")
        
        print(f"\n🔧 Next Steps:")
        print("1. Review the alignment results above")
        print("2. Listen to a few segments manually to verify")
        if quality['recommendation'] != 'redo':
            print("3. If acceptable, proceed with full segmentation")
        else:
            print("3. Consider forced alignment tools (Montreal Forced Alignment)")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Make sure you have the required dependencies installed")

if __name__ == "__main__":
    demo_alignment_verification() 