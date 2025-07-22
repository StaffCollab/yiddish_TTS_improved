#!/usr/bin/env python3
"""
Test Hebrew Word Order for TTS
Check if word sequence matches speaking order
"""

import json

print("🔍 Testing Hebrew Word Order for TTS")
print("=" * 50)

# Load sample
with open('tts_segments/segments_metadata.json', 'r') as f:
    metadata = json.load(f)

sample_text = metadata[0]['text']
print(f"Sample text: {sample_text}")
print()

# Split into words
words = sample_text.split()
print("📝 Word-by-word analysis:")
print("Text stored left-to-right:")
for i, word in enumerate(words[:8]):  # First 8 words
    print(f"  {i+1}: {word}")

print()
print("🎯 KEY QUESTION: Is this the speaking order?")
print()
print("🔍 In Hebrew/Yiddish:")
print("- Text is written RIGHT-TO-LEFT")
print("- But Unicode stores in LOGICAL order (reading order)")
print("- For TTS: Does logical order = speaking order?")
print()

# Check with a simple Hebrew sentence structure
print("📖 Analysis:")
print("Hebrew sentence: 'The boy ate the apple'")
print("Hebrew structure: [apple-the] [ate] [boy-the]")
print("Speaking order: boy-the → ate → apple-the")
print("Written order: apple-the ← ate ← boy-the (RTL)")
print("Stored order: [apple-the] [ate] [boy-the] (logical)")
print()

print("⚠️  POTENTIAL ISSUE:")
print("If stored left-to-right but spoken right-to-left,")
print("then word sequence might be backwards for TTS!")
print()

print("🔧 SOLUTION OPTIONS:")
print("1. Reverse word order: words.reverse()")
print("2. Check if transcription already matches audio order")
print("3. Test with a short sample to verify")
print()

print("🧪 Quick Test Needed:")
print("Listen to first audio file and check if word order matches:")
print("Audio says: [first_word] [second_word] [third_word]...")
print("Text shows:", " ".join(words[:6]), "...")
print()
print("If they don't match → need to reverse word order for TTS!") 