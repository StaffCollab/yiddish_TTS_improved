#!/usr/bin/env python3
"""
Yiddish Transliterator
Maps Hebrew characters to Latin equivalents for use with existing TTS models
This is a working solution while we set up proper Hebrew character training
"""

import os
import sys
import re
import torch
from TTS.api import TTS


class YiddishTransliterator:
    """Transliterates Yiddish (Hebrew script) to Latin characters"""
    
    def __init__(self):
        # Hebrew to Latin character mapping for Yiddish
        self.hebrew_to_latin = {
            # Basic Hebrew letters with Yiddish pronunciations
            'א': 'a',    # alef
            'ב': 'b',    # bet/vet
            'ג': 'g',    # gimel
            'ד': 'd',    # dalet
            'ה': 'h',    # heh
            'ו': 'u',    # vav (often u/v in Yiddish)
            'ז': 'z',    # zayin
            'ח': 'kh',   # chet
            'ט': 't',    # tet
            'י': 'i',    # yod
            'ך': 'kh',   # final kaf
            'כ': 'k',    # kaf
            'ל': 'l',    # lamed
            'ם': 'm',    # final mem
            'מ': 'm',    # mem
            'ן': 'n',    # final nun
            'נ': 'n',    # nun
            'ס': 's',    # samech
            'ע': 'e',    # ayin
            'ף': 'f',    # final peh
            'פ': 'p',    # peh
            'ץ': 'ts',   # final tsadi
            'צ': 'ts',   # tsadi
            'ק': 'k',    # qof
            'ר': 'r',    # resh
            'ש': 'sh',   # shin
            'ת': 't',    # tav
            
            # Vowel points (often used in Yiddish)
            'ַ': 'a',    # patach
            'ָ': 'o',    # qamats
            'ְ': '',     # shva (silent)
            'ֵ': 'e',    # tsere
            'ִ': 'i',    # hiriq
            'ֹ': 'o',    # holam
            'ּ': '',     # dagesh (ignore)
            
            # Punctuation - keep as is
            '.': '.', ',': ',', '!': '!', '?': '?', 
            ':': ':', ';': ';', '-': '-', ' ': ' ',
            '"': '"', "'": "'", '(': '(', ')': ')',
            '[': '[', ']': ']', '{': '{', '}': '}'
        }
        
        # Reverse mapping for converting back
        self.latin_to_hebrew = {v: k for k, v in self.hebrew_to_latin.items() if v}
    
    def transliterate_to_latin(self, yiddish_text):
        """Convert Yiddish (Hebrew script) to Latin characters"""
        result = []
        
        for char in yiddish_text:
            if char in self.hebrew_to_latin:
                result.append(self.hebrew_to_latin[char])
            elif char.isdigit() or char.isspace():
                result.append(char)
            else:
                # Keep unknown characters as is
                result.append(char)
        
        # Clean up multiple spaces and empty strings
        transliterated = ''.join(result)
        transliterated = re.sub(r'\s+', ' ', transliterated.strip())
        
        return transliterated
    
    def process_dataset(self, input_file, output_file):
        """Process entire dataset with transliteration"""
        
        print(f"Processing {input_file} -> {output_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        processed_lines = []
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                audio_file = parts[0]
                yiddish_text = parts[1]
                speaker_id = parts[2]
                
                # Transliterate the text
                latin_text = self.transliterate_to_latin(yiddish_text)
                
                # Create new line
                new_line = f"{audio_file}|{latin_text}|{speaker_id}"
                processed_lines.append(new_line)
        
        # Save processed dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"✓ Processed {len(processed_lines)} lines")
        return len(processed_lines)


def test_transliteration():
    """Test the transliteration with sample texts"""
    
    print("=== Testing Yiddish Transliteration ===")
    
    transliterator = YiddishTransliterator()
    
    # Test with your actual Yiddish texts
    test_texts = [
        "שבת שלום",
        "ווי גייט עס?",
        "געווען איז דאס פאריגע וואך מיטוואך",
        "א גוטן טאג און א גוטן יאר"
    ]
    
    for yiddish in test_texts:
        latin = transliterator.transliterate_to_latin(yiddish)
        print(f"Yiddish: {yiddish}")
        print(f"Latin:   {latin}")
        print()


def create_transliterated_dataset():
    """Create a transliterated version of your dataset"""
    
    print("=== Creating Transliterated Dataset ===")
    
    if not os.path.exists("yiddish_train_data.txt"):
        print("Error: yiddish_train_data.txt not found!")
        print("Please run: python prepare_yiddish_data.py")
        return False
    
    transliterator = YiddishTransliterator()
    
    # Process the dataset
    count = transliterator.process_dataset(
        "yiddish_train_data.txt",
        "yiddish_latin_train_data.txt"
    )
    
    print(f"✓ Created transliterated dataset: yiddish_latin_train_data.txt")
    print(f"  Contains {count} training samples")
    
    return True


def generate_yiddish_speech_transliterated(text, output_file="yiddish_latin_output.wav"):
    """Generate speech using transliterated text"""
    
    print("🎤 Generating Yiddish Speech (Transliterated)")
    
    transliterator = YiddishTransliterator()
    
    # Transliterate the input text
    latin_text = transliterator.transliterate_to_latin(text)
    
    print(f"Original: {text}")
    print(f"Transliterated: {latin_text}")
    
    try:
        # Use a simple English model that won't discard characters
        print("Loading TTS model...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        
        print("Generating speech...")
        tts.tts_to_file(text=latin_text, file_path=output_file)
        
        print(f"✓ Speech generated: {output_file}")
        print("🎉 Yiddish TTS working with transliteration!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("🔤 Yiddish Transliteration Solution")
    print("Converting Hebrew script to Latin for TTS compatibility")
    print()
    
    if len(sys.argv) > 1:
        # Generate speech for provided text
        text = " ".join(sys.argv[1:])
        generate_yiddish_speech_transliterated(text)
    else:
        # Run full process
        test_transliteration()
        
        if create_transliterated_dataset():
            print("\n=== Testing Speech Generation ===")
            test_text = "שבת שלום, ווי גייט עס?"
            generate_yiddish_speech_transliterated(test_text)
            
        print("\n=== Usage ===")
        print("Generate speech from Yiddish text:")
        print("python yiddish_transliterator.py 'שבת שלום'")
        print()
        print("This approach works immediately but pronunciation")
        print("may not be perfect. For best results, train from scratch") 