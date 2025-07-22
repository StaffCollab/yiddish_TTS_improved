#!/usr/bin/env python3
"""
Whisper Word Timing Analysis
Extract and display exact word timings from Whisper
"""

import whisper
import json
from pathlib import Path

def analyze_whisper_word_timings(audio_path, force_language=None):
    """Get detailed word timing analysis from Whisper"""
    print(f"🎤 WHISPER WORD TIMING ANALYSIS")
    print("=" * 50)
    print(f"Audio file: {Path(audio_path).name}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Transcribe with word timestamps
    if force_language:
        print(f"Forcing language: {force_language}")
        result = model.transcribe(audio_path, language=force_language, word_timestamps=True)
    else:
        print("Auto-detecting language...")
        result = model.transcribe(audio_path, word_timestamps=True)
    
    # Show basic info
    detected_language = result.get('language', 'unknown')
    full_text = result['text'].strip()
    
    print(f"\n📊 BASIC INFO:")
    print(f"   Detected language: {detected_language}")
    print(f"   Full transcription: {full_text[:100]}...")
    print(f"   Total text length: {len(full_text)} characters")
    
    # Extract all words with timings
    all_words = []
    
    for segment_idx, segment in enumerate(result["segments"]):
        segment_start = segment['start']
        segment_end = segment['end']
        segment_text = segment['text'].strip()
        
        print(f"\n🗣️ SEGMENT {segment_idx + 1}: {segment_start:.2f}s - {segment_end:.2f}s")
        print(f"   Text: {segment_text}")
        
        if "words" in segment:
            print(f"   Words ({len(segment['words'])}):")
            
            for word_idx, word in enumerate(segment["words"]):
                word_text = word['word'].strip()
                word_start = word['start']
                word_end = word['end']
                word_duration = word_end - word_start
                
                all_words.append({
                    'word': word_text,
                    'start': word_start,
                    'end': word_end,
                    'duration': word_duration,
                    'segment': segment_idx + 1
                })
                
                print(f"     {word_idx + 1:3d}. '{word_text}' -> {word_start:.2f}s - {word_end:.2f}s ({word_duration:.2f}s)")
        else:
            print("     No word-level timestamps available")
    
    # Summary statistics
    if all_words:
        total_words = len(all_words)
        avg_duration = sum(w['duration'] for w in all_words) / total_words
        min_duration = min(w['duration'] for w in all_words)
        max_duration = max(w['duration'] for w in all_words)
        total_speech_time = all_words[-1]['end'] - all_words[0]['start'] if all_words else 0
        
        print(f"\n📈 TIMING STATISTICS:")
        print(f"   Total words: {total_words}")
        print(f"   Speech duration: {total_speech_time:.1f}s")
        print(f"   Average word duration: {avg_duration:.3f}s")
        print(f"   Shortest word: {min_duration:.3f}s")
        print(f"   Longest word: {max_duration:.3f}s")
        print(f"   Speaking rate: {total_words / (total_speech_time/60):.1f} words/minute")
    
    return {
        'language': detected_language,
        'full_text': full_text,
        'segments': result['segments'],
        'words': all_words
    }

def save_timing_data(timing_data, output_file="whisper_timing_data.json"):
    """Save timing data to JSON for analysis"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timing_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Timing data saved: {output_file}")

def export_word_list(timing_data, output_file="whisper_word_list.txt"):
    """Export simple word list with timings"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WHISPER WORD TIMING LIST\n")
        f.write("=" * 40 + "\n\n")
        
        for i, word_data in enumerate(timing_data['words']):
            f.write(f"{i+1:3d}. {word_data['start']:7.2f}s - {word_data['end']:7.2f}s  '{word_data['word']}'\n")
    
    print(f"📝 Word list exported: {output_file}")

def compare_languages(audio_path):
    """Compare timing results with different language settings"""
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
                result = model.transcribe(audio_path, language=lang_code, word_timestamps=True)
            else:
                result = model.transcribe(audio_path, word_timestamps=True)
            
            # Count words
            word_count = 0
            for segment in result['segments']:
                if 'words' in segment:
                    word_count += len(segment['words'])
            
            results[lang_name] = {
                'detected_language': result.get('language', 'unknown'),
                'word_count': word_count,
                'text_preview': result['text'][:60] + "..." if result['text'] else ""
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
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        exit(1)
    
    print("🎯 WHISPER TIMING EXTRACTION")
    print("=" * 50)
    
    # First, compare different languages
    language_results = compare_languages(audio_path)
    
    # Then do detailed analysis (let's try auto-detect first)
    print(f"\n" + "="*60)
    timing_data = analyze_whisper_word_timings(audio_path)
    
    # Save the data
    save_timing_data(timing_data)
    export_word_list(timing_data)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Check whisper_timing_data.json for full data")
    print(f"   Check whisper_word_list.txt for word list")
    print(f"\n💡 Next step: Compare this to your original transcript")
    print(f"   Original transcript has: {len(open('original_files/transcripts/transcription1.txt').read().split())} words") 