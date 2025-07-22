#!/usr/bin/env python3
"""
Fix Hebrew Word Order for TTS
Handle Hebrew RTL directionality at word level
"""

import json
import unicodedata
import re

class HebrewAwareTokenizer:
    """Tokenizer that handles Hebrew word order correctly"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
    def build_vocab_from_texts(self, texts):
        """Build vocabulary from texts"""
        print("Building Hebrew-aware vocabulary...")
        
        all_chars = set()
        all_chars.update([self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN])
        
        for text in texts:
            normalized = self.normalize_text(text)
            all_chars.update(normalized)
        
        sorted_chars = sorted(list(all_chars))
        
        for idx, char in enumerate(sorted_chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        
        self.vocab_size = len(sorted_chars)
        print(f"✅ Vocabulary: {self.vocab_size} characters")
        
    def normalize_text(self, text):
        """Normalize Hebrew text"""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def is_hebrew_word(self, word):
        """Check if word contains Hebrew characters"""
        hebrew_chars = sum(1 for c in word if len(c) == 1 and 0x0590 <= ord(c) <= 0x05FF)
        return hebrew_chars > len(word) * 0.5  # More than 50% Hebrew chars
    
    def fix_word_order(self, text, reverse_hebrew_words=True):
        """Fix Hebrew word order for TTS"""
        if not reverse_hebrew_words:
            return text
            
        words = text.split()
        
        # Find Hebrew segments and reverse them
        fixed_words = []
        current_hebrew_segment = []
        
        for word in words:
            if self.is_hebrew_word(word):
                current_hebrew_segment.append(word)
            else:
                # Non-Hebrew word - process any pending Hebrew segment
                if current_hebrew_segment:
                    # Reverse the Hebrew segment for speaking order
                    fixed_words.extend(reversed(current_hebrew_segment))
                    current_hebrew_segment = []
                fixed_words.append(word)
        
        # Handle any remaining Hebrew segment
        if current_hebrew_segment:
            fixed_words.extend(reversed(current_hebrew_segment))
        
        return ' '.join(fixed_words)
    
    def text_to_sequence(self, text, reverse_hebrew_words=False):
        """Convert text to sequence with optional word order fix"""
        # Option to fix word order
        if reverse_hebrew_words:
            text = self.fix_word_order(text, True)
        
        normalized = self.normalize_text(text)
        sequence = [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) for char in normalized]
        return sequence


def test_word_order_fix():
    """Test the word order fixing"""
    print("🧪 Testing Hebrew Word Order Fix")
    print("=" * 50)
    
    # Load sample
    with open('tts_segments/segments_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    sample_text = metadata[0]['text'][:50]  # First part of sentence
    print(f"Original text: {sample_text}")
    
    tokenizer = HebrewAwareTokenizer()
    
    # Test both orders
    print("\n📝 Word Order Comparison:")
    
    # Original order
    words_original = sample_text.split()
    print(f"Original: {' | '.join(words_original[:6])}")
    
    # Fixed order (reversed Hebrew segments)  
    fixed_text = tokenizer.fix_word_order(sample_text, True)
    words_fixed = fixed_text.split()
    print(f"Fixed:    {' | '.join(words_fixed[:6])}")
    
    print("\n🎯 For TTS Training:")
    print("1. ORIGINAL ORDER: Use if transcription matches audio order")
    print("2. FIXED ORDER: Use if transcription is in visual Hebrew order")
    
    print("\n⚠️  CRITICAL: You need to verify which order matches your audio!")
    print("Listen to segment_0001.wav and check:")
    print(f"- Does it start with '{words_original[0]}'? → Use original")
    print(f"- Does it start with '{words_fixed[0]}'? → Use fixed")
    
    return tokenizer


if __name__ == "__main__":
    tokenizer = test_word_order_fix()
    
    print("\n🔧 Next Steps:")
    print("1. Listen to tts_segments/audio/segment_0001.wav")
    print("2. Check which word order matches the audio")
    print("3. Update training script with correct order") 