#!/usr/bin/env python3
"""
Yiddish Speech Generator
Uses XTTS v2 for zero-shot Yiddish speech synthesis
"""

import os
import sys
import torch
from TTS.api import TTS


def generate_yiddish_speech(text, output_file="yiddish_output.wav", reference_audio=None):
    """Generate Yiddish speech from text"""
    
    print("🎤 Yiddish Speech Generator")
    print("Using XTTS v2 for zero-shot voice cloning")
    print()
    
    # Default reference audio from your dataset
    if reference_audio is None:
        reference_audio = "tts_segments/audio/segment_0001.wav"
        
    if not os.path.exists(reference_audio):
        print(f"Error: Reference audio not found: {reference_audio}")
        print("Please make sure your audio files are in tts_segments/audio/")
        return
    
    print(f"Text to synthesize: {text}")
    print(f"Reference voice: {reference_audio}")
    print(f"Output file: {output_file}")
    print()
    
    try:
        print("Loading XTTS v2 model...")
        # Use XTTS v2 which is excellent for multilingual synthesis
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        print("✓ Model loaded!")
        
        print("Generating speech...")
        # Generate speech using Hebrew as the closest language to Yiddish
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=reference_audio,
            language="he"  # Hebrew is closest to Yiddish
        )
        
        print(f"✓ Speech generated: {output_file}")
        print()
        print("🎉 Success! Your Yiddish TTS is working!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_yiddish_phrases():
    """Test with common Yiddish phrases"""
    
    phrases = [
        "שבת שלום",  # Shabbat Shalom
        "ווי גייט עס?",  # How are you?
        "גוט מארגן",  # Good morning
        "א גוטן טאג",  # Good day
        "דאנק זייער",  # Thank you very much
        "געווען איז דאס פאריגע וואך מיטוואך"  # From your dataset
    ]
    
    print("Testing common Yiddish phrases:")
    print()
    
    output_dir = "yiddish_speech_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, phrase in enumerate(phrases, 1):
        output_file = os.path.join(output_dir, f"yiddish_phrase_{i}.wav")
        print(f"Phrase {i}: {phrase}")
        
        success = generate_yiddish_speech(phrase, output_file)
        if success:
            print(f"   ✓ Saved to: {output_file}")
        else:
            print(f"   ✗ Failed")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Generate speech for provided text
        text = " ".join(sys.argv[1:])
        generate_yiddish_speech(text)
    else:
        # Test with sample phrases
        print("No text provided. Testing with sample phrases...")
        print()
        test_yiddish_phrases()
        
        print("=== Usage ===")
        print("To generate speech for your own text:")
        print("python generate_yiddish_speech.py 'שבת שלום, ווי גייט עס?'")
        print()
        print("Your speech samples are in: yiddish_speech_samples/") 