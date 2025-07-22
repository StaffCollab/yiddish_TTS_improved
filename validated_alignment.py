#!/usr/bin/env python3
"""
Validated Yiddish Alignment
Use original transcript with Whisper timing validation via similarity scoring
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
import torchaudio
import warnings
warnings.filterwarnings("ignore")

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text segments"""
    # Simple character-level similarity for Hebrew/Yiddish
    text1_clean = ''.join(text1.split()).lower()
    text2_clean = ''.join(text2.split()).lower()
    
    if not text1_clean or not text2_clean:
        return 0.0
    
    # Character overlap ratio
    common_chars = set(text1_clean) & set(text2_clean)
    all_chars = set(text1_clean) | set(text2_clean)
    char_similarity = len(common_chars) / len(all_chars) if all_chars else 0.0
    
    # Length similarity (penalize very different lengths)
    len_ratio = min(len(text1_clean), len(text2_clean)) / max(len(text1_clean), len(text2_clean)) if max(len(text1_clean), len(text2_clean)) > 0 else 0.0
    
    # Word count similarity
    words1 = len(text1.split())
    words2 = len(text2.split())
    word_ratio = min(words1, words2) / max(words1, words2) if max(words1, words2) > 0 else 0.0
    
    # Combined similarity score
    similarity = (char_similarity * 0.4 + len_ratio * 0.3 + word_ratio * 0.3)
    
    return similarity

def create_validated_segments(audio_path, transcript_path, min_similarity=0.3):
    """Create segments with similarity validation"""
    import whisper
    
    print(f"🔍 VALIDATED ALIGNMENT for {Path(audio_path).name}")
    print(f"   Using original transcript with similarity validation")
    print(f"   Minimum similarity threshold: {min_similarity:.1f}")
    
    # Load original transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        original_transcript = f.read().strip()
    
    original_words = original_transcript.split()
    print(f"   📝 Original transcript: {len(original_words)} words")
    
    # Get Whisper Yiddish transcription with timings
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='yi', word_timestamps=True)
    
    # Extract Whisper words and timings
    whisper_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                whisper_words.append({
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end']
                })
    
    print(f"   🎤 Whisper detected: {len(whisper_words)} words")
    
    # Create candidate segments from original text with Whisper timings
    segments = []
    words_per_segment = 8  # Start with 8-word segments
    
    for i in range(0, len(original_words), words_per_segment):
        # Get original text segment
        segment_words = original_words[i:i + words_per_segment]
        segment_text = ' '.join(segment_words)
        
        # Get corresponding Whisper timing (approximate)
        start_word_idx = min(i, len(whisper_words) - 1)
        end_word_idx = min(i + len(segment_words) - 1, len(whisper_words) - 1)
        
        if start_word_idx < len(whisper_words) and end_word_idx < len(whisper_words):
            start_time = whisper_words[start_word_idx]['start']
            end_time = whisper_words[end_word_idx]['end']
            
            # Get what Whisper heard in this time range
            whisper_segment_words = []
            for w in whisper_words:
                if start_time <= w['start'] <= end_time:
                    whisper_segment_words.append(w['word'])
            
            whisper_segment_text = ' '.join(whisper_segment_words)
            
            # Calculate similarity
            similarity = calculate_text_similarity(segment_text, whisper_segment_text)
            
            segments.append({
                'original_text': segment_text,
                'whisper_text': whisper_segment_text,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'word_count': len(segment_words),
                'similarity': similarity,
                'is_valid': similarity >= min_similarity
            })
    
    # Filter valid segments
    valid_segments = [s for s in segments if s['is_valid']]
    invalid_segments = [s for s in segments if not s['is_valid']]
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Valid segments: {len(valid_segments)} (similarity ≥ {min_similarity:.1f})")
    print(f"   Invalid segments: {len(invalid_segments)} (similarity < {min_similarity:.1f})")
    
    if len(valid_segments) > 0:
        avg_similarity = np.mean([s['similarity'] for s in valid_segments])
        print(f"   Average similarity of valid segments: {avg_similarity:.3f}")
    
    # Show sample comparisons
    print(f"\n🔍 SAMPLE COMPARISONS:")
    for i, segment in enumerate(segments[:5]):
        status = "✅ VALID" if segment['is_valid'] else "❌ INVALID"
        print(f"\nSegment {i+1}: {status} (similarity: {segment['similarity']:.3f})")
        print(f"   Original : {segment['original_text'][:60]}...")
        print(f"   Whisper  : {segment['whisper_text'][:60]}...")
        print(f"   Duration : {segment['duration']:.1f}s")
    
    return valid_segments, invalid_segments

def extract_validated_segments(audio_path, valid_segments, output_dir="tts_segments_validated"):
    """Extract only the validated segments"""
    print(f"\n🎵 Extracting VALIDATED segments to {output_dir}/")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    (output_dir / "text").mkdir(exist_ok=True)
    
    # Load audio
    audio, sr = torchaudio.load(audio_path)
    
    extracted_segments = []
    
    for i, segment in enumerate(valid_segments):
        start_sample = int(segment['start'] * sr)
        end_sample = int(segment['end'] * sr)
        
        if start_sample < audio.shape[1] and end_sample <= audio.shape[1]:
            audio_segment = audio[:, start_sample:end_sample]
            
            # Save files
            segment_name = f"validated_segment_{i+1:04d}"
            audio_file = output_dir / "audio" / f"{segment_name}.wav"
            text_file = output_dir / "text" / f"{segment_name}.txt"
            
            torchaudio.save(str(audio_file), audio_segment, sr)
            
            # Use ORIGINAL text (not Whisper's)
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(segment['original_text'])
            
            extracted_segments.append({
                'audio_file': str(audio_file),
                'text_file': str(text_file),
                'text': segment['original_text'],
                'whisper_text': segment['whisper_text'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration'],
                'word_count': segment['word_count'],
                'similarity': segment['similarity']
            })
            
            print(f"   ✅ {audio_file.name} - {segment['duration']:.1f}s, similarity: {segment['similarity']:.3f}")
            print(f"      Text: {segment['original_text'][:50]}...")
    
    # Save metadata
    metadata_file = output_dir / "segments_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_segments, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Metadata saved: {metadata_file}")
    
    # Save validation report
    report_file = output_dir / "validation_report.json"
    validation_report = {
        'total_segments_created': len(valid_segments),
        'average_similarity': float(np.mean([s['similarity'] for s in valid_segments])),
        'similarity_threshold': 0.3,
        'segments': [
            {
                'segment_id': i+1,
                'similarity': s['similarity'],
                'duration': s['duration'],
                'word_count': s['word_count'],
                'original_text': s['original_text'],
                'whisper_heard': s['whisper_text']
            }
            for i, s in enumerate(valid_segments)
        ]
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    
    print(f"   📊 Validation report: {report_file}")
    
    return extracted_segments

def demo_validated_alignment():
    """Demo validated alignment approach"""
    print("🔍 VALIDATED YIDDISH ALIGNMENT")
    print("=" * 50)
    print("Keep original transcript + validate with similarity scoring")
    print("=" * 50)
    
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
        # Create validated segments
        valid_segments, invalid_segments = create_validated_segments(audio_path, transcript_path)
        
        if not valid_segments:
            print("\n❌ No valid segments found! Try lowering similarity threshold.")
            return
        
        # Extract validated segments
        extracted = extract_validated_segments(audio_path, valid_segments)
        
        print(f"\n✅ Validated alignment completed!")
        print(f"   Created {len(extracted)} validated segments")
        print(f"   Output directory: tts_segments_validated/")
        
        if extracted:
            print(f"\n🧪 Test the BEST segment:")
            best_segment = max(extracted, key=lambda x: x['similarity'])
            print(f"   Audio: {best_segment['audio_file']}")
            print(f"   Text: {best_segment['text']}")
            print(f"   Similarity: {best_segment['similarity']:.3f}")
            print(f"   Duration: {best_segment['duration']:.2f}s")
        
        print(f"\n🎯 WHY THIS SHOULD WORK:")
        print("   ✅ Uses your original high-quality transcript")
        print("   ✅ Validates alignment with similarity scoring")
        print("   ✅ Filters out poorly aligned segments")
        print("   ✅ Keeps only segments where text matches audio")
        
        print(f"\n🔧 Next Steps:")
        print("1. Test the best similarity segment")
        print("2. If good, process more audio files")
        print("3. Train TTS with validated segments")
        
    except Exception as e:
        print(f"❌ Validated alignment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_validated_alignment() 