#!/usr/bin/env python3
"""
Generate Word List Files for All Audio Files
Creates whisper_word_list.txt format for each audio file
"""

import whisper
import json
from pathlib import Path

def generate_word_list_for_audio(audio_path, output_file):
    """Generate word list with timing for a single audio file"""
    print(f"🎤 Processing {audio_path.name}...")
    
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Transcribe with word timestamps (auto-detect language for Latin transliterations)
    result = model.transcribe(str(audio_path), word_timestamps=True)
    
    # Extract word timings
    word_timings = []
    word_index = 1
    
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                word_timings.append({
                    'index': word_index,
                    'start': word['start'],
                    'end': word['end'],
                    'word': word['word'].strip()
                })
                word_index += 1
    
    print(f"   ✅ Found {len(word_timings)} words")
    
    # Write in the whisper_word_list.txt format
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"WHISPER WORD TIMING LIST\n")
        f.write(f"========================================\n\n")
        
        for timing in word_timings:
            f.write(f"{timing['index']:3d}.{timing['start']:8.2f}s -{timing['end']:8.2f}s  '{timing['word']}'\n")
    
    print(f"   💾 Saved to {output_file}")
    return len(word_timings)

def generate_all_word_lists(base_dir="original_files", output_dir="word_lists"):
    """Generate word lists for all audio files"""
    
    print("🎯 GENERATING WORD LISTS FOR ALL AUDIO FILES")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all audio files
    audio_dir = Path(base_dir) / "audio"
    audio_files = []
    
    for i in range(1, 22):  # audio1.wav through audio21.wav
        audio_file = audio_dir / f"audio{i}.wav"
        if audio_file.exists():
            audio_files.append((i, audio_file))
        else:
            print(f"⚠️  Missing audio file: {audio_file}")
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Check for existing word lists (checkpoint support)
    existing = []
    for i, audio_file in audio_files:
        word_list_file = output_path / f"audio{i}_word_list.txt"
        if word_list_file.exists():
            existing.append(i)
    
    if existing:
        print(f"🔄 Found {len(existing)} existing word lists: {existing}")
        print("   (Delete files in word_lists/ to regenerate)")
    
    # Generate missing word lists
    remaining = [(i, af) for i, af in audio_files if i not in existing]
    print(f"⏳ Generating {len(remaining)} word lists...")
    
    results = []
    
    for idx, (i, audio_file) in enumerate(remaining, 1):
        print(f"\n[{idx}/{len(remaining)}] Processing audio{i}.wav...")
        
        try:
            output_file = output_path / f"audio{i}_word_list.txt"
            word_count = generate_word_list_for_audio(audio_file, output_file)
            
            results.append({
                'audio_id': f"audio{i}",
                'audio_file': str(audio_file),
                'word_list_file': str(output_file),
                'word_count': word_count,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'audio_id': f"audio{i}",
                'audio_file': str(audio_file),
                'error': str(e),
                'status': 'failed'
            })
    
    # Save results summary
    summary_file = output_path / "generation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(audio_files),
            'existing': len(existing),
            'generated': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    # Final summary
    success_count = len([r for r in results if r['status'] == 'success'])
    total_completed = len(existing) + success_count
    
    print(f"\n🎉 WORD LIST GENERATION COMPLETE!")
    print(f"   Total audio files: {len(audio_files)}")
    print(f"   Previously completed: {len(existing)}")
    print(f"   Newly generated: {success_count}")
    print(f"   Total completed: {total_completed}")
    print(f"   Output directory: {output_path}")
    print(f"   Summary: {summary_file}")

def main():
    generate_all_word_lists()

if __name__ == "__main__":
    main() 