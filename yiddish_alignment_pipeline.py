#!/usr/bin/env python3
"""
Yiddish TTS Alignment Pipeline
Processes all 21 audio/transcript pairs to create perfect aligned segments
Combines whisper_transcribe.py + whisper_word_mapper.py functionality
"""

import json
import re
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Tuple
import librosa
import soundfile as sf

# Import whisper functionality
try:
    import whisper
    from faster_whisper import WhisperModel
    import whisperx
    WHISPER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Whisper imports failed: {e}")
    print("    Pipeline will attempt basic whisper only")
    WHISPER_AVAILABLE = False

class YiddishAlignmentPipeline:
    def __init__(self, base_dir="original_files", output_dir="aligned_dataset"):
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.transcript_dir = self.base_dir / "transcripts"
        self.output_dir = Path(output_dir)
        
        # Create output structure
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "segments").mkdir(exist_ok=True)
        (self.output_dir / "timings").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Load whisper model once
        self.whisper_model = None
        self.load_whisper_model()
    
    def load_whisper_model(self):
        """Load Whisper model for transcription"""
        try:
            print("🤖 Loading Whisper model...")
            
            # Try to force CUDA if available
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"   🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print(f"   💻 Using CPU (CUDA not available)")
            
            self.whisper_model = whisper.load_model("base", device=device)
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            print("   Will attempt fallback methods")
    
    def get_audio_transcript_pairs(self) -> List[Tuple[Path, Path]]:
        """Get all matching audio/transcript pairs"""
        pairs = []
        
        for i in range(1, 22):  # audio1.wav through audio21.wav
            audio_file = self.audio_dir / f"audio{i}.wav"
            transcript_file = self.transcript_dir / f"transcription{i}.txt"
            
            if audio_file.exists() and transcript_file.exists():
                pairs.append((audio_file, transcript_file))
            else:
                print(f"⚠️  Missing pair {i}: audio={audio_file.exists()}, transcript={transcript_file.exists()}")
        
        print(f"📁 Found {len(pairs)} complete audio/transcript pairs")
        return pairs
    
    def transcribe_with_timing(self, audio_path: Path) -> List[Dict]:
        """Get word-level timing from Whisper"""
        
        if not self.whisper_model:
            raise Exception("Whisper model not available")
        
        print(f"   🎤 Transcribing {audio_path.name}...")
        
        try:
            # Try with word timestamps first
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="yi",  # Force Yiddish
                word_timestamps=True
            )
            
            # Extract word-level timings
            timings = []
            word_idx = 1
            
            for segment in result.get('segments', []):
                for word_data in segment.get('words', []):
                    timings.append({
                        'index': word_idx,
                        'start': word_data['start'],
                        'end': word_data['end'],
                        'whisper_word': word_data['word'].strip(),
                        'duration': word_data['end'] - word_data['start']
                    })
                    word_idx += 1
            
            return timings
            
        except Exception as e:
            print(f"   ⚠️  Word timestamps failed: {e}")
            print(f"   🔄 Falling back to segment-based timing...")
            
            # Fallback: use segments and estimate word timing
            result = self.whisper_model.transcribe(str(audio_path), language="yi")
            timings = []
            word_idx = 1
            
            for segment in result.get('segments', []):
                words = segment['text'].strip().split()
                segment_start = segment['start']
                segment_end = segment['end']
                segment_duration = segment_end - segment_start
                
                if len(words) > 0:
                    word_duration = segment_duration / len(words)
                    
                    for i, word in enumerate(words):
                        word_start = segment_start + (i * word_duration)
                        word_end = word_start + word_duration
                        
                        timings.append({
                            'index': word_idx,
                            'start': word_start,
                            'end': word_end,
                            'whisper_word': word.strip(),
                            'duration': word_duration
                        })
                        word_idx += 1
            
            return timings
    
    def load_original_transcript(self, transcript_path: Path) -> List[str]:
        """Load original Yiddish transcript"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        words = text.split()
        return words
    
    def create_word_mapping(self, original_words: List[str], whisper_timings: List[Dict]) -> List[Dict]:
        """Map original words to whisper timings"""
        
        original_count = len(original_words)
        timing_count = len(whisper_timings)
        
        print(f"   🔗 Mapping: {original_count} original words → {timing_count} timings")
        
        mapped_words = []
        
        # Proportional mapping
        for i in range(original_count):
            timing_idx = min(int(i * timing_count / original_count), timing_count - 1)
            timing = whisper_timings[timing_idx]
            
            mapped_words.append({
                'original_word': original_words[i],
                'whisper_word': timing['whisper_word'],
                'start': timing['start'],
                'end': timing['end'],
                'duration': timing['duration'],
                'original_index': i + 1,
                'timing_index': timing['index']
            })
        
        return mapped_words
    
    def create_segments(self, mapped_words: List[Dict], words_per_segment=8, end_buffer=1.0) -> List[Dict]:
        """Create TTS training segments with generous buffer"""
        
        segments = []
        current_segment = []
        segment_id = 1
        
        for word_data in mapped_words:
            current_segment.append(word_data)
            
            if len(current_segment) >= words_per_segment:
                segment_text = ' '.join(w['original_word'] for w in current_segment)
                segment_start = current_segment[0]['start']
                segment_end = current_segment[-1]['end'] + end_buffer
                
                segments.append({
                    'segment_id': f"{segment_id:04d}",
                    'start': segment_start,
                    'end': segment_end,
                    'duration': segment_end - segment_start,
                    'text': segment_text,
                    'word_count': len(current_segment),
                    'words': current_segment
                })
                
                current_segment = []
                segment_id += 1
        
        # Handle remaining words
        if current_segment:
            segment_text = ' '.join(w['original_word'] for w in current_segment)
            segment_start = current_segment[0]['start']
            segment_end = current_segment[-1]['end'] + end_buffer
            
            segments.append({
                'segment_id': f"{segment_id:04d}",
                'start': segment_start,
                'end': segment_end,
                'duration': segment_end - segment_start,
                'text': segment_text,
                'word_count': len(current_segment),
                'words': current_segment
            })
        
        return segments
    
    def export_segments(self, segments: List[Dict], audio_path: Path, file_id: str):
        """Export aligned segments for one audio file"""
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=16000)
        
        # Create file-specific directories
        audio_output_dir = self.output_dir / "segments" / f"file_{file_id}" / "audio"
        text_output_dir = self.output_dir / "segments" / f"file_{file_id}" / "text"
        audio_output_dir.mkdir(parents=True, exist_ok=True)
        text_output_dir.mkdir(parents=True, exist_ok=True)
        
        segments_metadata = []
        
        for seg in segments:
            segment_id = seg['segment_id']
            
            # Export text
            text_file = text_output_dir / f"segment_{segment_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(seg['text'])
            
            # Export audio
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            end_sample = min(end_sample, len(audio))  # Clamp to audio length
            
            if start_sample < len(audio):
                audio_segment = audio[start_sample:end_sample]
                audio_file = audio_output_dir / f"segment_{segment_id}.wav"
                sf.write(str(audio_file), audio_segment, sr)
            
            segments_metadata.append({
                'segment_id': segment_id,
                'start_time': seg['start'],
                'end_time': seg['end'],
                'duration': seg['duration'],
                'text': seg['text'],
                'word_count': seg['word_count'],
                'audio_file': f"segments/file_{file_id}/audio/segment_{segment_id}.wav",
                'text_file': f"segments/file_{file_id}/text/segment_{segment_id}.txt"
            })
        
        # Save file metadata
        metadata_file = self.output_dir / "metadata" / f"file_{file_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'file_id': file_id,
                'source_audio': str(audio_path),
                'total_segments': len(segments_metadata),
                'total_words': sum(seg['word_count'] for seg in segments),
                'segments': segments_metadata
            }, f, indent=2, ensure_ascii=False)
        
        return segments_metadata
    
    def process_single_pair(self, audio_path: Path, transcript_path: Path) -> Dict:
        """Process one audio/transcript pair"""
        
        file_id = audio_path.stem  # e.g., "audio1"
        print(f"\n📄 Processing {file_id}...")
        
        try:
            # Step 1: Get word timings from Whisper
            whisper_timings = self.transcribe_with_timing(audio_path)
            print(f"   ⏱️  Got {len(whisper_timings)} word timings")
            
            # Step 2: Load original transcript
            original_words = self.load_original_transcript(transcript_path)
            print(f"   📝 Loaded {len(original_words)} original words")
            
            # Step 3: Create word mapping
            mapped_words = self.create_word_mapping(original_words, whisper_timings)
            
            # Step 4: Create segments
            segments = self.create_segments(mapped_words)
            print(f"   📦 Created {len(segments)} segments")
            
            # Step 5: Export everything
            segments_metadata = self.export_segments(segments, audio_path, file_id)
            
            # Save timing data for debugging
            timing_file = self.output_dir / "timings" / f"{file_id}_timings.json"
            with open(timing_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'file_id': file_id,
                    'whisper_timings': whisper_timings,
                    'mapped_words': mapped_words
                }, f, indent=2, ensure_ascii=False)
            
            return {
                'file_id': file_id,
                'status': 'success',
                'segments': len(segments),
                'words': len(original_words),
                'duration': max(seg['end_time'] for seg in segments_metadata)
            }
            
        except Exception as e:
            print(f"   ❌ Error processing {file_id}: {e}")
            traceback.print_exc()
            return {
                'file_id': file_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_pipeline(self):
        """Run the complete pipeline on all pairs"""
        
        print("🚀 YIDDISH TTS ALIGNMENT PIPELINE")
        print("=" * 50)
        
        pairs = self.get_audio_transcript_pairs()
        
        if not pairs:
            print("❌ No audio/transcript pairs found!")
            return
        
        results = []
        total_segments = 0
        total_words = 0
        
        for i, (audio_path, transcript_path) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] Processing pair...")
            
            result = self.process_single_pair(audio_path, transcript_path)
            results.append(result)
            
            if result['status'] == 'success':
                total_segments += result['segments']
                total_words += result['words']
        
        # Save overall results
        pipeline_results = {
            'total_files': len(pairs),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'total_segments': total_segments,
            'total_words': total_words,
            'results': results
        }
        
        results_file = self.output_dir / "pipeline_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(pipeline_results, f, indent=2, ensure_ascii=False)
        
        # Final summary
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"   Total files: {len(pairs)}")
        print(f"   Successful: {pipeline_results['successful']}")
        print(f"   Failed: {pipeline_results['failed']}")
        print(f"   Total segments: {total_segments}")
        print(f"   Total words: {total_words}")
        print(f"   Output directory: {self.output_dir}")

def main():
    pipeline = YiddishAlignmentPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 