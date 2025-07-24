#!/usr/bin/env python3
"""
Forced Alignment Solution for Yiddish Text and Audio
Properly align known text with audio using multiple approaches
"""

import whisper
import json
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import find_peaks
import re
from difflib import SequenceMatcher
from typing import List, Dict

class YiddishAudioTextAligner:
    def __init__(self, audio_path, transcript_path):
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.audio = None
        self.sr = None
        self.original_text = None
        self.load_data()
    
    def load_data(self):
        """Load audio and text data"""
        print("📁 Loading audio and text data...")
        
        # Load audio
        self.audio, self.sr = librosa.load(self.audio_path, sr=16000)
        print(f"   Audio: {len(self.audio)/self.sr:.1f}s at {self.sr}Hz")
        
        # Load text
        with open(self.transcript_path, 'r', encoding='utf-8') as f:
            self.original_text = f.read().strip()
        
        words = self.original_text.split()
        print(f"   Text: {len(words)} words, {len(self.original_text)} characters")
    
    def detect_silence_segments(self, min_silence_duration=0.5, silence_threshold=0.01):
        """Detect silence-based natural breakpoints in audio"""
        print(f"\n🔇 DETECTING SILENCE-BASED SEGMENTS")
        print("=" * 40)
        
        # Calculate RMS energy
        frame_length = int(0.025 * self.sr)  # 25ms frames
        hop_length = int(0.01 * self.sr)     # 10ms hop
        
        rms = librosa.feature.rms(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to time
        times = librosa.frames_to_time(range(len(rms)), sr=self.sr, hop_length=hop_length)
        
        # Find silence (low energy regions)
        silence_mask = rms < silence_threshold
        
        # Find continuous silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_start = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                silence_end = times[i]
                silence_duration = silence_end - silence_start
                if silence_duration >= min_silence_duration:
                    silence_regions.append((silence_start, silence_end, silence_duration))
                in_silence = False
        
        print(f"Found {len(silence_regions)} silence regions:")
        for i, (start, end, duration) in enumerate(silence_regions):
            print(f"  {i+1:2d}. {start:6.2f}s - {end:6.2f}s ({duration:.2f}s)")
        
        return silence_regions
    
    def get_whisper_timing_structure(self):
        """Get Whisper's segment timing (ignore the text, keep the timing)"""
        print(f"\n🎤 EXTRACTING WHISPER TIMING STRUCTURE")
        print("=" * 40)
        
        model = whisper.load_model("base")
        
        # Try multiple languages to get best timing structure
        languages_to_try = [None, 'de', 'yi', 'he']  # Auto, German, Yiddish, Hebrew
        best_result = None
        best_score = 0
        
        for lang in languages_to_try:
            try:
                if lang:
                    result = model.transcribe(self.audio_path, language=lang)
                else:
                    result = model.transcribe(self.audio_path)
                
                # Score based on number of segments (more segments = better timing resolution)
                score = len(result['segments'])
                print(f"  Language {lang or 'auto'}: {score} segments")
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    
            except Exception as e:
                print(f"  Language {lang}: Error - {e}")
        
        if not best_result:
            raise Exception("Could not get timing from Whisper")
        
        timing_segments = []
        for seg in best_result['segments']:
            timing_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['end'] - seg['start']
            })
        
        print(f"Selected timing structure: {len(timing_segments)} segments")
        return timing_segments
    
    def create_smart_text_segments(self, target_segment_count):
        """Intelligently split text into approximately target_segment_count parts"""
        print(f"\n📝 CREATING SMART TEXT SEGMENTS")
        print("=" * 40)
        
        words = self.original_text.split()
        total_words = len(words)
        
        print(f"Total words: {total_words}")
        print(f"Target segments: {target_segment_count}")
        print(f"Words per segment (avg): {total_words/target_segment_count:.1f}")
        
        # Find natural breakpoints (punctuation)
        text_with_positions = []
        word_index = 0
        
        for word in words:
            text_with_positions.append((word, word_index))
            word_index += 1
        
        # Look for natural breakpoints
        breakpoints = []
        sentences = re.split(r'[.!?،؛]+', self.original_text)
        
        current_word_count = 0
        for sentence in sentences:
            sentence_words = sentence.strip().split()
            if sentence_words:
                current_word_count += len(sentence_words)
                breakpoints.append(current_word_count)
        
        print(f"Natural breakpoints at words: {breakpoints}")
        
        # If we have too many or too few natural breakpoints, adjust
        if len(breakpoints) > target_segment_count * 1.5:
            # Too many breakpoints, use every nth one
            step = len(breakpoints) // target_segment_count
            breakpoints = breakpoints[::step]
        elif len(breakpoints) < target_segment_count * 0.7:
            # Too few breakpoints, add artificial ones
            words_per_segment = total_words // target_segment_count
            for i in range(1, target_segment_count):
                artificial_breakpoint = i * words_per_segment
                breakpoints.append(artificial_breakpoint)
            breakpoints = sorted(set(breakpoints))
        
        # Create text segments
        text_segments = []
        start_word = 0
        
        for breakpoint in breakpoints:
            if breakpoint <= total_words:
                segment_words = words[start_word:breakpoint]
                if segment_words:
                    text_segments.append({
                        'words': segment_words,
                        'text': ' '.join(segment_words),
                        'word_start': start_word,
                        'word_end': breakpoint,
                        'word_count': len(segment_words)
                    })
                start_word = breakpoint
        
        # Add final segment if there are remaining words
        if start_word < total_words:
            final_words = words[start_word:]
            text_segments.append({
                'words': final_words,
                'text': ' '.join(final_words),
                'word_start': start_word,
                'word_end': total_words,
                'word_count': len(final_words)
            })
        
        print(f"Created {len(text_segments)} text segments:")
        for i, seg in enumerate(text_segments):
            print(f"  {i+1:2d}. Words {seg['word_start']:3d}-{seg['word_end']:3d} ({seg['word_count']:2d} words): {seg['text'][:50]}...")
        
        return text_segments
    
    def distribute_words_by_time(self, timing_segments, punctuation_window=5):
        """Slice the transcript so that each timing segment gets a word chunk
        whose length is proportional to its duration.  Optionally snap the
        cut-point to the nearest punctuation within ±punctuation_window words."""
        total_duration = sum(seg['duration'] for seg in timing_segments)
        words = self.original_text.split()
        total_words = len(words)
        avg_rate = total_words / total_duration  # words per second
 
        allocated_segments = []
        word_cursor = 0
 
        for idx, seg in enumerate(timing_segments):
            if idx == len(timing_segments) - 1:
                # Last segment gets whatever words remain
                word_slice = words[word_cursor:]
            else:
                raw_count = seg['duration'] * avg_rate
                target_count = int(round(raw_count))
 
                # Keep at least 1 word per segment
                target_count = max(1, target_count)
 
                # Make sure we don't run past the end when rounding errors add up
                if word_cursor + target_count >= total_words:
                    target_count = total_words - word_cursor - 1
 
                # Try to snap to punctuation if possible
                slice_end = word_cursor + target_count
                search_start = max(word_cursor + 1, slice_end - punctuation_window)
                search_end   = min(total_words - 1, slice_end + punctuation_window)
 
                for j in range(search_start, search_end):
                    if re.match(r'.*[,:;.!?—-]$', words[j]):
                        slice_end = j + 1  # include the punctuation word
                        break
 
                word_slice = words[word_cursor:slice_end]
                word_cursor = slice_end
 
            text_chunk = ' '.join(word_slice)
            allocated_segments.append({
                'segment_id': idx + 1,
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg['duration'],
                'text': text_chunk,
                'word_count': len(word_slice),
                'words': word_slice
            })
 
        return allocated_segments
 
    def allocate_windowed(self, timing_segments, window_size=4, punctuation_window=5):
        """Allocate words to timing segments using a moving speaking-rate window.
        This greatly reduces cumulative drift because the rate is recomputed
        every <window_size> segments instead of once for the whole file."""
        words = self.original_text.split()
        total_words = len(words)
        total_duration = sum(seg['duration'] for seg in timing_segments)
 
        segments_out = []
        word_idx = 0  # pointer into transcript words list
        # average seconds per word (use 10% safety buffer)
        avg_word_sec = (total_duration / total_words) * 1.1 if total_words else 0.5
 
        # Helper to build the segment dict
        def make_seg(timing_seg, slice_words, seg_id):
            return {
                'segment_id': seg_id,
                'start': timing_seg['start'],
                'end': timing_seg['end'],
                'duration': timing_seg['duration'],
                'text': ' '.join(slice_words),
                'word_count': len(slice_words),
                'words': slice_words
            }
 
        for i, tseg in enumerate(timing_segments):
            if i == len(timing_segments) - 1:
                # Last segment gets the rest
                slice_words = words[word_idx:]
                segments_out.append(make_seg(tseg, slice_words, i + 1))
                break
 
            # ------------------------
            # 1. Estimate local rate
            # ------------------------
            lookahead = timing_segments[i: i + window_size]
            lookahead_dur = sum(s['duration'] for s in lookahead)
            remaining_words = total_words - word_idx
            local_rate = remaining_words / lookahead_dur if lookahead_dur > 0 else 0
 
            # ------------------------
            # 2. How many words fit?
            # ------------------------
            target_count = int(round(tseg['duration'] * local_rate))
            target_count = max(1, target_count)

            # Clamp by duration so ultra-short clips don't get too many words
            if tseg['duration'] < 0.6:
                target_count = 1
            else:
                max_allowed = max(1, int(tseg['duration'] / avg_word_sec))
                target_count = min(target_count, max_allowed)
 
            if word_idx + target_count >= total_words - 1:
                target_count = total_words - word_idx - 1
 
            slice_end = word_idx + target_count
            search_start = max(word_idx + 1, slice_end - punctuation_window)
            search_end   = min(total_words - 1, slice_end + punctuation_window)
 
            for j in range(search_start, search_end):
                if re.search(r'[,:;.!?—-]$', words[j]):
                    slice_end = j + 1
                    break
 
            slice_words = words[word_idx: slice_end]
            segments_out.append(make_seg(tseg, slice_words, i + 1))
            word_idx = slice_end
 
        return segments_out
 
    def get_silence_based_timing_segments(self, min_silence_duration: float = 0.5, silence_threshold: float = 0.01) -> List[Dict]:
        """Build timing segments using the RMS-energy based silence detector we already have.
        This avoids the heavy librosa.effects dependency that drags in pandas/scikit-learn.
        """
        print(f"\n🔇 BUILDING TIMING FROM SIMPLE SILENCE DETECTION")

        silence_regions = self.detect_silence_segments(min_silence_duration=min_silence_duration,
                                                       silence_threshold=silence_threshold)

        total_dur = len(self.audio) / self.sr
        segments: List[Dict] = []

        # Build non-silent spans between the detected silences
        cursor = 0.0
        for (sil_start, sil_end, _) in silence_regions:
            if sil_start - cursor >= 0.1:  # at least 100 ms of speech
                segments.append({
                    'start': cursor,
                    'end': sil_start,
                    'duration': sil_start - cursor
                })
            cursor = sil_end  # skip over the silence

        # tail speech after last silence
        if total_dur - cursor >= 0.1:
            segments.append({
                'start': cursor,
                'end': total_dur,
                'duration': total_dur - cursor
            })

        print(f"Found {len(segments)} speech segments via RMS silence")
        return segments

    def create_aligned_segments(self, use_silence: bool = True):
        """Main method to create properly aligned segments"""
        print(f"🎯 YIDDISH AUDIO-TEXT ALIGNMENT")
        print("=" * 50)

        # Step 1: Choose timing structure
        if use_silence:
            timing_segments = self.get_silence_based_timing_segments()
            # If silence produces too few segments, fall back to Whisper
            if len(timing_segments) < 10:
                print("⚠️  Silence detection produced too few segments – falling back to Whisper timing")
                timing_segments = self.get_whisper_timing_structure()
        else:
            timing_segments = self.get_whisper_timing_structure()
        
        # Step 2: Distribute words to each timing segment based on duration
        aligned_segments = self.allocate_windowed(timing_segments, window_size=4)
        
        # Step 3: Detect silence for optional validation
        silence_regions = self.detect_silence_segments()
        
        return {
            'aligned_segments': aligned_segments,
            'timing_segments': timing_segments,
            'silence_regions': silence_regions,
            'total_duration': aligned_segments[-1]['end'] if aligned_segments else 0,
            'total_words': sum(seg['word_count'] for seg in aligned_segments)
        }
    
    def export_segments(self, alignment_result, output_dir="yiddish_aligned_segments"):
        """Export aligned segments to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create audio and text directories
        (output_path / "audio").mkdir(exist_ok=True)
        (output_path / "text").mkdir(exist_ok=True)
        
        segments_metadata = []
        
        for seg in alignment_result['aligned_segments']:
            segment_id = f"{seg['segment_id']:04d}"
            
            # Export text
            text_file = output_path / "text" / f"yiddish_segment_{segment_id}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(seg['text'])
            
            # Export audio segment
            start_sample = int(seg['start'] * self.sr)
            end_sample = int(seg['end'] * self.sr)
            audio_segment = self.audio[start_sample:end_sample]
            
            audio_file = output_path / "audio" / f"yiddish_segment_{segment_id}.wav"
            import soundfile as sf
            sf.write(audio_file, audio_segment, self.sr)
            
            # Metadata
            segments_metadata.append({
                'segment_id': segment_id,
                'start_time': seg['start'],
                'end_time': seg['end'],
                'duration': seg['duration'],
                'text': seg['text'],
                'word_count': seg['word_count'],
                'audio_file': f"audio/yiddish_segment_{segment_id}.wav",
                'text_file': f"text/yiddish_segment_{segment_id}.txt"
            })
        
        # Save metadata
        metadata_file = output_path / "segments_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_segments': len(segments_metadata),
                'total_duration': alignment_result['total_duration'],
                'total_words': alignment_result['total_words'],
                'segments': segments_metadata
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 SEGMENTS EXPORTED")
        print(f"   Directory: {output_path}")
        print(f"   Audio files: {len(segments_metadata)}")
        print(f"   Text files: {len(segments_metadata)}")
        print(f"   Metadata: segments_metadata.json")
        
        return output_path

def main():
    audio_path = "original_files/audio/audio1.wav"
    transcript_path = "original_files/transcripts/transcription1.txt"
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    if not Path(transcript_path).exists():
        print(f"❌ Transcript file not found: {transcript_path}")
        return
    
    # Create aligner
    aligner = YiddishAudioTextAligner(audio_path, transcript_path)
    
    # Create alignment
    alignment_result = aligner.create_aligned_segments()
    
    # Export segments
    output_dir = aligner.export_segments(alignment_result)
    
    print(f"\n✅ ALIGNMENT COMPLETE!")
    print(f"   Created {len(alignment_result['aligned_segments'])} aligned segments")
    print(f"   Total words: {alignment_result['total_words']}")
    print(f"   Total duration: {alignment_result['total_duration']:.1f}s")
    print(f"   Output: {output_dir}")

if __name__ == "__main__":
    main() 