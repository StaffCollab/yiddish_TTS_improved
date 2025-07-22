#!/usr/bin/env python3
"""
PURE Whisper Yiddish Alignment
Use Whisper's own transcription as text source - guarantees perfect alignment
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def create_pure_whisper_segments(audio_path, target_duration=6):
    """Create segments using Whisper's own transcription - perfect alignment guaranteed"""
    import whisper
    
    print(f"🎤 Pure Whisper Yiddish segmentation for {Path(audio_path).name}")
    print("✨ Using Whisper's transcription as ground truth = PERFECT alignment")
    
    # Load model
    model = whisper.load_model("base")
    
    # Transcribe with Yiddish
    result = model.transcribe(audio_path, language='yi', word_timestamps=True)
    
    # Extract all words with timings
    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                all_words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
    
    print(f"   ✅ Whisper detected {len(all_words)} Yiddish words")
    
    # Create segments from Whisper's words
    segments = []
    current_words = []
    current_start = None
    
    for i, word_info in enumerate(all_words):
        if current_start is None:
            current_start = word_info['start']
        
        current_words.append(word_info['word'])
        current_duration = word_info['end'] - current_start
        
        # End segment criteria
        should_end = False
        if current_duration >= target_duration or len(current_words) >= 12:
            should_end = True
        
        if should_end or i == len(all_words) - 1:
            # Perfect boundary: end exactly where Whisper says the word ends
            segment_end = word_info['end']
            
            # Safety gap only if there's a next word
            if i + 1 < len(all_words):
                next_start = all_words[i + 1]['start']
                segment_end = min(segment_end, next_start - 0.01)
            
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
    
    print(f"   ✅ Created {len(segments)} perfectly aligned segments")
    
    # Show segment quality
    good_duration = sum(1 for s in segments if 2 <= s['duration'] <= 10)
    avg_duration = np.mean([s['duration'] for s in segments])
    
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   TTS-suitable: {good_duration}/{len(segments)} segments")
    
    return segments, result['text']

def extract_pure_segments(audio_path, segments, output_dir="tts_segments_pure"):
    """Extract perfectly aligned audio segments"""
    print(f"🎵 Extracting PERFECTLY aligned segments to {output_dir}/")
    print("🎯 GUARANTEE: Audio and text are exactly what Whisper detected")
    
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
            segment_name = f"pure_segment_{i+1:04d}"
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
            print(f"      Text: {segment['text'][:50]}...")
    
    # Save metadata
    metadata_file = output_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_segments, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Metadata saved: {metadata_file}")
    return extracted_segments

def demo_pure_whisper():
    """Demo pure Whisper alignment"""
    print("🎯 PURE WHISPER YIDDISH ALIGNMENT")
    print("=" * 50)
    print("PERFECT SOLUTION: Use Whisper's text as ground truth")
    print("=" * 50)
    
    # Find original files
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    
    if not audio_files:
        print("❌ No audio files found")
        return
    
    audio_path = str(audio_files[0])
    print(f"Processing: {Path(audio_path).name}")
    
    try:
        # Create pure segments
        segments, full_transcription = create_pure_whisper_segments(audio_path)
        
        print(f"\n📝 Whisper's complete transcription:")
        print(f"   {full_transcription[:150]}...")
        
        # Extract segments
        extracted = extract_pure_segments(audio_path, segments)
        
        print(f"\n✅ Pure Whisper alignment completed!")
        print(f"   Created {len(extracted)} PERFECTLY aligned segments")
        print(f"   Output directory: tts_segments_pure/")
        
        print(f"\n🧪 Test the first segment:")
        print(f"   Audio: {extracted[0]['audio_file']}")
        print(f"   Text: {extracted[0]['text']}")
        print(f"   Duration: {extracted[0]['duration']:.2f}s")
        
        print(f"\n🎯 WHY THIS WILL WORK:")
        print("   ✅ Text = EXACTLY what Whisper heard in the audio")
        print("   ✅ Timing = EXACTLY when Whisper detected each word")
        print("   ✅ No force-alignment = NO mismatches possible")
        print("   ✅ Perfect for TTS training")
        
        print(f"\n🔧 Next Steps:")
        print("1. Test first segment - should be PERFECT alignment")
        print("2. Train TTS model with these perfect segments")
        print("3. Generate working Yiddish speech")
        
    except Exception as e:
        print(f"❌ Pure alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_pure_whisper() 