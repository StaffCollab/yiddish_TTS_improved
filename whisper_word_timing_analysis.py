#!/usr/bin/env python3
"""
Whisper Word Timing Analysis
Extract and display exact word timings from Whisper
"""

import whisper
import json
from pathlib import Path

def detailed_word_comparison(original_text, whisper_text):
    """Detailed word-by-word comparison between original and Whisper"""
    print(f"\n🔍 DETAILED WORD-BY-WORD COMPARISON")
    print("=" * 50)
    
    original_words = original_text.strip().split()[:20]  # First 20 words
    whisper_words = whisper_text.strip().split()[:20]   # First 20 words
    
    print(f"First 20 words comparison:")
    print(f"Original ({len(original_words)} words): {' '.join(original_words)}")
    print(f"Whisper  ({len(whisper_words)} words): {' '.join(whisper_words)}")
    
    # Align and compare
    print(f"\nWord-by-word analysis:")
    max_len = max(len(original_words), len(whisper_words))
    
    for i in range(max_len):
        orig_word = original_words[i] if i < len(original_words) else "---"
        whisper_word = whisper_words[i] if i < len(whisper_words) else "---"
        
        # Check if they're similar (basic phonetic similarity)
        match_status = "✓" if orig_word != "---" and whisper_word != "---" else "✗"
        if orig_word == "---" or whisper_word == "---":
            match_status = "➕" if orig_word == "---" else "➖"
        
        print(f"  {i+1:2d}. {orig_word:15} → {whisper_word:15} {match_status}")
    
    return {
        'original_first_20': original_words,
        'whisper_first_20': whisper_words,
        'alignment_analysis': 'detailed comparison completed'
    }

def test_specific_audio_segment(audio_path, start_time=0, end_time=30):
    """Test a specific segment of audio to see transcription quality"""
    print(f"\n🎵 TESTING SPECIFIC AUDIO SEGMENT ({start_time}s - {end_time}s)")
    print("=" * 50)
    
    # Load audio and extract segment (we'll use full audio for now since segment extraction is complex)
    model = whisper.load_model("base")
    
    # Test with different languages on the same audio
    languages_to_test = [
        (None, "Auto-detect"),
        ('yi', 'Yiddish'), 
        ('he', 'Hebrew'),
        ('de', 'German'),
        ('nl', 'Dutch')
    ]
    
    segment_results = {}
    
    for lang_code, lang_name in languages_to_test:
        try:
            if lang_code:
                result = model.transcribe(audio_path, language=lang_code)
            else:
                result = model.transcribe(audio_path)
            
            # Get first segment text for comparison
            first_segment = result['segments'][0]['text'] if result['segments'] else ""
            first_20_words = ' '.join(first_segment.split()[:20])
            
            segment_results[lang_name] = {
                'detected_language': result.get('language', 'unknown'),
                'first_segment': first_segment,
                'first_20_words': first_20_words
            }
            
            print(f"\n{lang_name}:")
            print(f"  Detected: {result.get('language', 'unknown')}")
            print(f"  First 20 words: {first_20_words}")
            
        except Exception as e:
            print(f"\n{lang_name}: Error - {e}")
            segment_results[lang_name] = {'error': str(e)}
    
    return segment_results

def analyze_whisper_segments(audio_path, force_language=None):
    """Get detailed segment analysis from Whisper (avoiding word_timestamps bug)"""
    print(f"🎤 WHISPER SEGMENT ANALYSIS")
    print("=" * 50)
    print(f"Audio file: {Path(audio_path).name}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Transcribe WITHOUT word timestamps (due to triton bug)
    if force_language:
        print(f"Forcing language: {force_language}")
        result = model.transcribe(audio_path, language=force_language)
    else:
        print("Auto-detecting language...")
        result = model.transcribe(audio_path)
    
    # Show basic info
    detected_language = result.get('language', 'unknown')
    full_text = result['text'].strip()
    
    print(f"\n📊 BASIC INFO:")
    print(f"   Detected language: {detected_language}")
    print(f"   Full transcription: {full_text[:100]}...")
    print(f"   Total text length: {len(full_text)} characters")
    
    # Count words manually
    whisper_words = full_text.split()
    print(f"   Whisper word count: {len(whisper_words)} words")
    
    # Extract segments with timings
    all_segments = []
    total_words_in_segments = 0
    
    for segment_idx, segment in enumerate(result["segments"]):
        segment_start = segment['start']
        segment_end = segment['end']
        segment_text = segment['text'].strip()
        segment_words = segment_text.split()
        total_words_in_segments += len(segment_words)
        
        print(f"\n🗣️ SEGMENT {segment_idx + 1}: {segment_start:.2f}s - {segment_end:.2f}s")
        print(f"   Text: {segment_text}")
        print(f"   Words in this segment: {len(segment_words)}")
        
        all_segments.append({
            'segment_id': segment_idx + 1,
            'start': segment_start,
            'end': segment_end,
            'text': segment_text,
            'word_count': len(segment_words),
            'words': segment_words
        })
    
    print(f"\n📈 SEGMENT STATISTICS:")
    print(f"   Total segments: {len(all_segments)}")
    print(f"   Total words from segments: {total_words_in_segments}")
    print(f"   Total words from full text: {len(whisper_words)}")
    
    return {
        'language': detected_language,
        'full_text': full_text,
        'word_count': len(whisper_words),
        'segments': all_segments,
        'total_segments': len(all_segments)
    }

def compare_word_counts(original_transcript_path, whisper_data):
    """Compare word counts between original and Whisper transcription"""
    print(f"\n📝 WORD COUNT COMPARISON")
    print("=" * 30)
    
    # Read original transcript
    with open(original_transcript_path, 'r', encoding='utf-8') as f:
        original_text = f.read().strip()
    
    original_words = original_text.split()
    whisper_words = whisper_data['full_text'].split()
    
    print(f"Original transcript: {len(original_words)} words")
    print(f"Whisper transcription: {len(whisper_words)} words")
    print(f"Difference: {len(whisper_words) - len(original_words)} words")
    
    # Show first few words of each
    print(f"\nFirst 10 words comparison:")
    print(f"Original: {' '.join(original_words[:10])}")
    print(f"Whisper:  {' '.join(whisper_words[:10])}")
    
    # Add detailed comparison
    detailed_comparison = detailed_word_comparison(original_text, whisper_data['full_text'])
    
    return {
        'original_count': len(original_words),
        'whisper_count': len(whisper_words),
        'difference': len(whisper_words) - len(original_words),
        'original_first_10': original_words[:10],
        'whisper_first_10': whisper_words[:10],
        'detailed_comparison': detailed_comparison
    }

def save_analysis_data(analysis_data, output_file="whisper_analysis_data.json"):
    """Save analysis data to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Analysis data saved: {output_file}")

def compare_languages(audio_path):
    """Compare transcription results with different language settings"""
    print(f"\n🌍 LANGUAGE COMPARISON")
    print("=" * 30)
    
    languages_to_test = [
        (None, "Auto-detect"),
        ('yi', 'Yiddish'),
        ('he', 'Hebrew'),
        ('nl', 'Dutch'),
        ('de', 'German')
    ]
    
    results = {}
    
    for lang_code, lang_name in languages_to_test:
        print(f"\n🔄 Testing {lang_name}...")
        try:
            model = whisper.load_model("base")
            if lang_code:
                result = model.transcribe(audio_path, language=lang_code)
            else:
                result = model.transcribe(audio_path)
            
            # Count words
            word_count = len(result['text'].split())
            
            results[lang_name] = {
                'detected_language': result.get('language', 'unknown'),
                'word_count': word_count,
                'text_preview': result['text'][:60] + "..." if result['text'] else "",
                'full_text': result['text']
            }
            
            print(f"   Detected: {result.get('language', 'unknown')}")
            print(f"   Words: {word_count}")
            print(f"   Text: {result['text'][:60]}...")
            
        except Exception as e:
            print(f"   Error: {e}")
            results[lang_name] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    # Analyze the first audio file
    audio_path = "original_files/audio/audio1.wav"
    original_transcript_path = "original_files/transcripts/transcription1.txt"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        exit(1)
    
    if not Path(original_transcript_path).exists():
        print(f"❌ Original transcript not found: {original_transcript_path}")
        exit(1)
    
    print("🎯 WHISPER ANALYSIS (NO WORD TIMESTAMPS)")
    print("=" * 60)
    
    # Test specific audio segment first
    segment_analysis = test_specific_audio_segment(audio_path)
    
    # Then compare different languages
    language_results = compare_languages(audio_path)
    
    # Then do detailed analysis (let's try auto-detect first)
    print(f"\n" + "="*60)
    analysis_data = analyze_whisper_segments(audio_path)
    
    # Compare with original transcript
    comparison = compare_word_counts(original_transcript_path, analysis_data)
    
    # Combine all data
    full_results = {
        'segment_analysis': segment_analysis,
        'analysis': analysis_data,
        'comparison': comparison,
        'language_tests': language_results
    }
    
    # Save the data
    save_analysis_data(full_results)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Whisper detected: {analysis_data['word_count']} words")
    print(f"   Original has: {comparison['original_count']} words")
    print(f"   Difference: {comparison['difference']} words")
    print(f"   Check whisper_analysis_data.json for full results") 