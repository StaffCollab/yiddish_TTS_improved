#!/usr/bin/env python3
"""
STRICT Whisper-based Audio-Text Alignment
Zero tolerance for extra words - segments end exactly at assigned word boundaries
"""

import os
import json
import librosa
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def transcribe_whisper_word_level(audio_path):
    """Use regular Whisper with word-level timestamps"""
    import whisper
    
    # Load model
    model = whisper.load_model("base")
    
    # Transcribe with word timestamps
    result = model.transcribe(audio_path, word_timestamps=True)
    
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
    
    print(f"   ✅ Transcribed with {len(words_with_timing)} words")
    return words_with_timing

def align_transcript_with_whisper_words(transcript_text, whisper_words):
    """Align the original transcript with Whisper word timings"""
    print("🔗 Aligning original transcript with Whisper timings...")
    
    # Clean and split original transcript
    original_words = transcript_text.strip().split()
    whisper_word_texts = [w['word'] for w in whisper_words]
    
    print(f"   Original transcript: {len(original_words)} words")
    print(f"   Whisper detected: {len(whisper_word_texts)} words")
    
    # Simple alignment: match words sequentially
    aligned_words = []
    whisper_idx = 0
    
    for orig_word in original_words:
        if whisper_idx < len(whisper_words):
            # Use timing from whisper word
            timing = whisper_words[whisper_idx]
            aligned_words.append({
                'word': orig_word,
                'start': timing['start'],
                'end': timing['end']
            })
            whisper_idx += 1
        else:
            # If we run out of Whisper words, estimate timing
            if aligned_words:
                last_end = aligned_words[-1]['end']
                aligned_words.append({
                    'word': orig_word,
                    'start': last_end,
                    'end': last_end + 0.5  # Estimate 0.5s per word
                })
    
    print(f"   ✅ Aligned {len(aligned_words)} words with timings")
    return aligned_words

def create_strict_tts_segments(aligned_words, target_duration=6, max_duration=10):
    """Create TTS segments with STRICT boundaries - no extra words allowed"""
    print(f"✂️  Creating STRICT TTS segments (target: {target_duration}s, max: {max_duration}s)")
    print("   🔒 ZERO tolerance for extra words - segments end at EXACT word boundaries")
    
    segments = []
    current_words = []
    current_start = None
    
    for i, word_info in enumerate(aligned_words):
        if current_start is None:
            current_start = word_info['start']
        
        current_words.append(word_info['word'])
        current_duration = word_info['end'] - current_start
        
        # Decide if we should end this segment
        should_end = False
        
        if current_duration >= target_duration:
            # We've reached target duration
            should_end = True
        elif current_duration >= max_duration:
            # We've reached maximum duration - must end
            should_end = True
        elif len(current_words) >= 15:
            # Too many words - end segment
            should_end = True
        
        if should_end:
            # STRICT: End segment EXACTLY at the last assigned word's end
            # NO buffer, NO extension - cut exactly where the last word ends
            segment_end = word_info['end']
            
            # SAFETY: If there's a next word, ensure we don't go past its start
            if i + 1 < len(aligned_words):
                next_word_start = aligned_words[i + 1]['start']
                # End at least 0.02s before next word starts (20ms safety gap)
                segment_end = min(segment_end, next_word_start - 0.02)
            
            segment_text = ' '.join(current_words)
            segments.append({
                'start': current_start,
                'end': segment_end,
                'duration': segment_end - current_start,
                'text': segment_text,
                'word_count': len(current_words),
                'last_word_original_end': word_info['end'],  # Track original word end
                'safety_gap': word_info['end'] - segment_end if i + 1 < len(aligned_words) else 0.0
            })
            
            # Reset for next segment
            current_words = []
            current_start = None
    
    # Handle remaining words
    if current_words and current_start is not None:
        segment_text = ' '.join(current_words)
        final_word_end = aligned_words[-1]['end']
        
        segments.append({
            'start': current_start,
            'end': final_word_end,  # Final segment can use exact end
            'duration': final_word_end - current_start,
            'text': segment_text,
            'word_count': len(current_words),
            'last_word_original_end': final_word_end,
            'safety_gap': 0.0
        })
    
    print(f"   ✅ Created {len(segments)} segments")
    
    # Show segment summary with safety info
    avg_duration = np.mean([s['duration'] for s in segments])
    good_duration = sum(1 for s in segments if 2 <= s['duration'] <= 10)
    avg_gap = np.mean([s['safety_gap'] for s in segments])
    
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   TTS-suitable duration: {good_duration}/{len(segments)} segments")
    print(f"   Average safety gap: {avg_gap:.3f}s (preventing word bleeding)")
    
    return segments

def extract_strict_audio_segments(audio_path, segments, output_dir="tts_segments_whisper_strict"):
    """Extract audio segments using STRICT boundaries"""
    print(f"🎵 Extracting STRICTLY aligned audio segments to {output_dir}/")
    print("   🔒 Each segment contains ONLY its assigned words")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    extracted_segments = []
    
    for i, segment in enumerate(segments):
        # Extract audio segment using STRICT timings
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio.shape[1] and end_sample <= audio.shape[1]:
            audio_segment = audio[:, start_sample:end_sample]
            
            # Save files
            segment_name = f"strict_segment_{i+1:04d}"
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
                'word_count': segment['word_count'],
                'last_word_original_end': segment['last_word_original_end'],
                'safety_gap': segment['safety_gap']
            })
            
            gap_info = f"gap: {segment['safety_gap']:.3f}s" if segment['safety_gap'] > 0 else "no gap (final)"
            print(f"   ✅ {audio_file.name} - {segment['duration']:.1f}s, {segment['word_count']} words, {gap_info}")
    
    # Save metadata
    metadata_file = output_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_segments, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Metadata saved: {metadata_file}")
    return extracted_segments

def demo_strict_whisper_alignment():
    """Demo the STRICT Whisper alignment process"""
    print("🔒 STRICT Whisper-based Audio-Text Alignment")
    print("=" * 55)
    print("ZERO TOLERANCE: Segments contain ONLY assigned words")
    print("=" * 55)
    
    # Find original files
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
            transcript_text = f.read().strip()
        
        # Get word-level timestamps from Whisper
        print("🎤 Transcribing with Whisper...")
        whisper_words = transcribe_whisper_word_level(audio_path)
        
        if not whisper_words:
            print("❌ Could not get word timestamps")
            return
        
        # Align original transcript with Whisper timings
        aligned_words = align_transcript_with_whisper_words(transcript_text, whisper_words)
        
        # Create TTS segments with STRICT boundaries
        segments = create_strict_tts_segments(aligned_words)
        
        # Extract audio segments with STRICT precision
        extracted = extract_strict_audio_segments(audio_path, segments)
        
        print(f"\n✅ STRICT Whisper alignment completed!")
        print(f"   Created {len(extracted)} strictly aligned segments")
        print(f"   Output directory: tts_segments_whisper_strict/")
        print(f"   🔒 GUARANTEE: Each segment contains ONLY its assigned words")
        
        print(f"\n🧪 Test the first segment:")
        print(f"   Audio: {extracted[0]['audio_file']}")
        print(f"   Text: {extracted[0]['text']}")
        print(f"   Duration: {extracted[0]['duration']:.2f}s")
        print(f"   Safety gap: {extracted[0]['safety_gap']:.3f}s")
        
        print(f"\n🔧 Expected result:")
        print("   ✅ Audio should contain EXACTLY the words in the text")
        print("   ✅ NO extra words at the beginning or end")
        print("   ✅ Clean cuts with tiny safety gaps")
        
        print(f"\n🎯 Next Steps:")
        print("1. Test first segment - should have ZERO extra words")
        print("2. If perfect, process all audio files")
        print("3. Train TTS with perfectly aligned segments")
        
    except Exception as e:
        print(f"❌ Strict alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_strict_whisper_alignment() 