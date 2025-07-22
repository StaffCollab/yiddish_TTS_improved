#!/usr/bin/env python3
"""
Test Whisper with Hebrew Language Forcing
See if audio is actually Hebrew/Yiddish but Whisper detected wrong language
"""

import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

def test_whisper_languages():
    """Test Whisper with different language settings"""
    print("🌍 WHISPER LANGUAGE TEST")
    print("=" * 50)
    print("Testing if audio is Hebrew/Yiddish but detected as Dutch")
    print("=" * 50)
    
    # Find audio file
    audio_files = sorted(Path("original_files/audio").glob("audio*.wav"))[:1]
    if not audio_files:
        print("❌ No audio files found")
        return
    
    audio_path = str(audio_files[0])
    print(f"📁 Testing: {Path(audio_path).name}")
    
    import whisper
    model = whisper.load_model("base")
    
    # Test 1: Auto-detect (what we saw before)
    print(f"\n🔄 Test 1: Auto-detect language")
    result_auto = model.transcribe(audio_path, word_timestamps=True)
    detected_lang = result_auto.get('language', 'unknown')
    auto_text = result_auto['text'][:100]
    print(f"   Detected language: {detected_lang}")
    print(f"   Transcription: {auto_text}...")
    
    # Test 2: Force Hebrew
    print(f"\n🔄 Test 2: Force Hebrew language")
    try:
        result_hebrew = model.transcribe(audio_path, language='he', word_timestamps=True)
        hebrew_text = result_hebrew['text'][:100]
        print(f"   Hebrew transcription: {hebrew_text}...")
    except Exception as e:
        print(f"   ❌ Hebrew failed: {e}")
    
    # Test 3: Force Yiddish (if available)
    print(f"\n🔄 Test 3: Force Yiddish language")
    try:
        result_yiddish = model.transcribe(audio_path, language='yi', word_timestamps=True)
        yiddish_text = result_yiddish['text'][:100]
        print(f"   Yiddish transcription: {yiddish_text}...")
    except Exception as e:
        print(f"   ❌ Yiddish not available: {e}")
    
    # Test 4: Try German (close to Yiddish)
    print(f"\n🔄 Test 4: Force German language")
    try:
        result_german = model.transcribe(audio_path, language='de', word_timestamps=True)
        german_text = result_german['text'][:100]
        print(f"   German transcription: {german_text}...")
    except Exception as e:
        print(f"   ❌ German failed: {e}")
    
    # Load original transcript for comparison
    transcript_files = sorted(Path("original_files/transcripts").glob("transcription*.txt"))[:1]
    if transcript_files:
        with open(transcript_files[0], 'r', encoding='utf-8') as f:
            original_text = f.read().strip()[:100]
        print(f"\n📝 Original transcript: {original_text}...")
    
    print(f"\n🎯 ANALYSIS:")
    print("Compare the transcriptions above:")
    print("1. Does Hebrew transcription match the original better?")
    print("2. Does any forced language make more sense?")
    print("3. Or is the audio actually in Dutch/English?")
    
    # Show available languages
    print(f"\n🌍 Available Whisper languages:")
    languages = ['en', 'he', 'yi', 'de', 'nl', 'fr', 'es', 'ar']
    for lang in languages:
        try:
            test_result = model.transcribe(audio_path, language=lang, word_timestamps=False)
            sample = test_result['text'][:50]
            print(f"   {lang}: {sample}...")
        except:
            print(f"   {lang}: Not available")
        
        if lang == 'nl':  # Stop after testing a few
            break

if __name__ == "__main__":
    test_whisper_languages() 