#!/usr/bin/env python3
"""
Whisper-based Audio-Text Alignment
Use Whisper to get word-level timings for proper segmentation
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

def get_word_level_timestamps():
    """Use Whisper to get word-level timestamps"""
    print("🤖 Setting up Whisper for word-level alignment...")
    
    try:
        # Try to use whisperx for word-level timestamps
        import whisperx
        
        print("   ✅ Using WhisperX for precise word timestamps")
        return "whisperx"
        
    except ImportError:
        try:
            # Fallback to regular whisper with word timestamps
            import whisper
            
            print("   ✅ Using Whisper with word timestamps")
            return "whisper"
            
        except ImportError:
            print("   ❌ Neither WhisperX nor Whisper installed")
            print("   📦 Install with: pip install whisperx")
            print("   📦 Or: pip install openai-whisper")
            return None

def transcribe_with_word_timestamps(audio_path, method="whisper"):
    """Get transcription with word-level timestamps"""
    print(f"🎤 Transcribing {Path(audio_path).name} with word timestamps...")
    
    if method == "whisperx":
        return transcribe_whisperx(audio_path)
    elif method == "whisper":
        return transcribe_whisper_word_level(audio_path)
    else:
        return None

def transcribe_whisperx(audio_path):
    """Use WhisperX for precise alignment"""
    import whisperx
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16" if device == "cuda" else "int8"
    
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("base", device, compute_type=compute_type)
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    print(f"   ✅ Transcribed with {len(result['segments'])} segments")
    
    # Extract word-level timestamps
    words_with_timing = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                if "start" in word and "end" in word:
                    words_with_timing.append({
                        'word': word['word'].strip(),
                        'start': word['start'],
                        'end': word['end']
                    })
    
    return words_with_timing

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
    # This is not perfect but better than proportional timing
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

def create_tts_segments_from_word_timings(aligned_words, target_duration=6, max_duration=10):
    """Create TTS segments from word-level timings"""
    print(f"✂️  Creating TTS segments (target: {target_duration}s, max: {max_duration}s)")
    
    segments = []
    current_words = []
    current_start = None
    
    for word_info in aligned_words:
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
            # Create segment
            segment_text = ' '.join(current_words)
            segments.append({
                'start': current_start,
                'end': word_info['end'],
                'duration': word_info['end'] - current_start,
                'text': segment_text,
                'word_count': len(current_words)
            })
            
            # Reset for next segment
            current_words = []
            current_start = None
    
    # Handle remaining words
    if current_words and current_start is not None:
        segment_text = ' '.join(current_words)
        final_end = aligned_words[-1]['end']
        segments.append({
            'start': current_start,
            'end': final_end,
            'duration': final_end - current_start,
            'text': segment_text,
            'word_count': len(current_words)
        })
    
    print(f"   ✅ Created {len(segments)} segments")
    
    # Show segment summary
    avg_duration = np.mean([s['duration'] for s in segments])
    good_duration = sum(1 for s in segments if 2 <= s['duration'] <= 10)
    
    print(f"   Average duration: {avg_duration:.1f}s")
    print(f"   TTS-suitable duration: {good_duration}/{len(segments)} segments")
    
    return segments

def extract_aligned_audio_segments(audio_path, segments, output_dir="tts_segments_whisper"):
    """Extract audio segments using precise timings"""
    print(f"🎵 Extracting aligned audio segments to {output_dir}/")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    extracted_segments = []
    
    for i, segment in enumerate(segments):
        # Extract audio segment using precise timings
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio.shape[1] and end_sample <= audio.shape[1]:
            audio_segment = audio[:, start_sample:end_sample]
            
            # Save files
            segment_name = f"whisper_segment_{i+1:04d}"
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

def demo_whisper_alignment():
    """Demo the Whisper alignment process"""
    print("🎤 Whisper-based Audio-Text Alignment Demo")
    print("=" * 50)
    
    # Check Whisper availability
    method = get_word_level_timestamps()
    if not method:
        return
    
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
        whisper_words = transcribe_with_word_timestamps(audio_path, method)
        
        if not whisper_words:
            print("❌ Could not get word timestamps")
            return
        
        # Align original transcript with Whisper timings
        aligned_words = align_transcript_with_whisper_words(transcript_text, whisper_words)
        
        # Create TTS segments
        segments = create_tts_segments_from_word_timings(aligned_words)
        
        # Extract audio segments
        extracted = extract_aligned_audio_segments(audio_path, segments)
        
        print(f"\n✅ Whisper alignment completed!")
        print(f"   Created {len(extracted)} precisely aligned segments")
        print(f"   Output directory: tts_segments_whisper/")
        
        print(f"\n🔧 Next Steps:")
        print("1. Test a few segments manually to verify alignment")
        print("2. If alignment is good, process all your audio files")
        print("3. Use these segments for TTS training")
        
    except Exception as e:
        print(f"❌ Whisper alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_whisper_alignment() 