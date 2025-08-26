#!/usr/bin/env python3
"""
Convert Aegisub .ass subtitle file to TTS training data
Extracts audio segments with precise timing alignment
"""

import os
import re
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def parse_ass_time(time_str: str) -> float:
    """Convert ASS timestamp format (H:MM:SS.CC) to seconds"""
    # Format: 0:00:01.88
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    return hours * 3600 + minutes * 60 + seconds

def parse_ass_file(ass_file_path: str) -> List[Dict]:
    """Parse .ass file and extract dialogue entries with timestamps and text"""
    entries = []
    
    with open(ass_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all dialogue lines (ignore comments)
    dialogue_pattern = r'Dialogue: \d+,([^,]+),([^,]+),[^,]*,[^,]*,\d+,\d+,\d+,[^,]*,(.+)'
    
    for match in re.finditer(dialogue_pattern, content):
        start_time_str = match.group(1)
        end_time_str = match.group(2)
        text = match.group(3).strip()
        
        # Skip empty text
        if not text:
            continue
            
        start_time = parse_ass_time(start_time_str)
        end_time = parse_ass_time(end_time_str)
        duration = end_time - start_time
        
        # Skip very short segments (less than 0.3 seconds)
        if duration < 0.3:
            continue
            
        entries.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'text': text,
            'start_time_str': start_time_str,
            'end_time_str': end_time_str
        })
    
    # Sort by start time
    entries.sort(key=lambda x: x['start_time'])
    
    print(f"📊 Parsed {len(entries)} dialogue entries from .ass file")
    return entries

def extract_audio_segment(audio_data: np.ndarray, sample_rate: int, 
                         start_time: float, end_time: float,
                         method: str = "precise") -> np.ndarray:
    """Extract audio segment using different methods"""
    
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    if method == "precise":
        # Exact frame extraction
        segment = audio_data[start_sample:end_sample]
        
    elif method == "padded":
        # Add small padding (50ms on each side)
        padding_samples = int(0.05 * sample_rate)
        start_sample = max(0, start_sample - padding_samples)
        end_sample = min(len(audio_data), end_sample + padding_samples)
        segment = audio_data[start_sample:end_sample]
        
    elif method == "crossfade":
        # Add padding with crossfade
        padding_samples = int(0.1 * sample_rate)
        fade_samples = int(0.02 * sample_rate)  # 20ms fade
        
        start_sample = max(0, start_sample - padding_samples)
        end_sample = min(len(audio_data), end_sample + padding_samples)
        segment = audio_data[start_sample:end_sample]
        
        # Apply fade in/out
        if len(segment) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            segment[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            segment[-fade_samples:] *= fade_out
            
    elif method == "silence_trimmed":
        # Extract with padding, then trim silence
        padding_samples = int(0.1 * sample_rate)
        start_sample = max(0, start_sample - padding_samples)
        end_sample = min(len(audio_data), end_sample + padding_samples)
        segment = audio_data[start_sample:end_sample]
        
        # Trim silence using librosa
        segment_trimmed, _ = librosa.effects.trim(segment, top_db=30)
        segment = segment_trimmed if len(segment_trimmed) > 0 else segment
        
    else:
        raise ValueError(f"Unknown extraction method: {method}")
    
    return segment

def normalize_audio(audio_segment: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio segment to target RMS level"""
    current_rms = np.sqrt(np.mean(audio_segment**2))
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        # Prevent clipping
        scaling_factor = min(scaling_factor, 1.0 / np.max(np.abs(audio_segment)))
        return audio_segment * scaling_factor
    return audio_segment

def create_training_data(ass_file_path: str, audio_file_path: str, 
                        output_dir: str, methods: List[str] = None):
    """Main function to create training data from .ass file"""
    
    if methods is None:
        methods = ["precise", "padded", "crossfade", "silence_trimmed"]
    
    print(f"🎵 Loading audio file: {audio_file_path}")
    
    # Load audio
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)
    print(f"📊 Audio loaded: {len(audio_data)} samples at {sample_rate}Hz ({len(audio_data)/sample_rate:.2f}s)")
    
    # Parse .ass file
    entries = parse_ass_file(ass_file_path)
    
    # Create output directories for each method
    base_output_dir = Path(output_dir)
    
    training_data_lines = {}
    metadata = {}
    
    for method in methods:
        method_dir = base_output_dir / f"ass_segments_{method}"
        audio_dir = method_dir / "audio"
        text_dir = method_dir / "text"
        
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)
        
        training_data_lines[method] = []
        metadata[method] = {
            'method': method,
            'source_ass': ass_file_path,
            'source_audio': audio_file_path,
            'sample_rate': sample_rate,
            'segments': []
        }
        
        print(f"\n🔧 Processing with method: {method}")
        
        for i, entry in enumerate(entries):
            segment_id = f"ass_segment_{i+1:04d}"
            
            # Extract audio segment
            audio_segment = extract_audio_segment(
                audio_data, sample_rate, 
                entry['start_time'], entry['end_time'], 
                method=method
            )
            
            # Skip if segment is too short after processing
            if len(audio_segment) < int(0.3 * sample_rate):
                print(f"⚠️  Skipping segment {segment_id}: too short after processing")
                continue
            
            # Normalize audio
            audio_segment = normalize_audio(audio_segment)
            
            # Save audio file
            audio_filename = f"{segment_id}.wav"
            audio_path = audio_dir / audio_filename
            sf.write(audio_path, audio_segment, sample_rate)
            
            # Save text file
            text_filename = f"{segment_id}.txt"
            text_path = text_dir / text_filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(entry['text'])
            
            # Add to training data
            relative_audio_path = f"ass_segments_{method}/audio/{audio_filename}"
            training_line = f"{relative_audio_path}|{entry['text']}|speaker_0"
            training_data_lines[method].append(training_line)
            
            # Add to metadata
            metadata[method]['segments'].append({
                'segment_id': segment_id,
                'start_time': entry['start_time'],
                'end_time': entry['end_time'],
                'duration': entry['duration'],
                'actual_duration': len(audio_segment) / sample_rate,
                'text': entry['text'],
                'text_length': len(entry['text']),
                'audio_path': str(audio_path),
                'text_path': str(text_path)
            })
            
            if (i + 1) % 10 == 0:
                print(f"  ✅ Processed {i + 1}/{len(entries)} segments")
        
        # Save training data file
        training_file = base_output_dir / f"yiddish_ass_{method}_train_data.txt"
        with open(training_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(training_data_lines[method]))
        
        # Save metadata
        metadata_file = method_dir / f"{method}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata[method], f, indent=2, ensure_ascii=False)
        
        print(f"  📁 Created {len(training_data_lines[method])} training samples")
        print(f"  💾 Saved: {training_file}")
        print(f"  📊 Metadata: {metadata_file}")
    
    # Create summary
    print(f"\n📈 SUMMARY:")
    print(f"  Source: {ass_file_path}")
    print(f"  Audio: {audio_file_path}")
    print(f"  Output: {output_dir}")
    
    for method in methods:
        total_duration = sum(seg['actual_duration'] for seg in metadata[method]['segments'])
        avg_duration = total_duration / len(metadata[method]['segments']) if metadata[method]['segments'] else 0
        avg_text_length = sum(seg['text_length'] for seg in metadata[method]['segments']) / len(metadata[method]['segments']) if metadata[method]['segments'] else 0
        
        print(f"  {method:15}: {len(metadata[method]['segments']):3d} segments, "
              f"{total_duration:6.1f}s total, {avg_duration:4.1f}s avg, "
              f"{avg_text_length:4.1f} chars avg")
    
    # Create a recommended config file
    create_fastspeech2_config(base_output_dir, methods[0], sample_rate, metadata[methods[0]])
    
    return metadata

def create_fastspeech2_config(output_dir: Path, best_method: str, sample_rate: int, metadata: Dict):
    """Create a FastSpeech2 config optimized for this data"""
    
    # Analyze the data
    segments = metadata['segments']
    total_duration = sum(seg['actual_duration'] for seg in segments)
    avg_text_length = sum(seg['text_length'] for seg in segments) / len(segments)
    
    # Extract unique characters
    all_text = ' '.join(seg['text'] for seg in segments)
    unique_chars = sorted(set(all_text))
    characters_string = ''.join(unique_chars)
    
    config = {
        "model": "fastspeech2",
        "run_name": f"yiddish_ass_{best_method}",
        "run_description": f"Yiddish TTS trained on manually aligned .ass data using {best_method} extraction",
        
        # Dataset
        "datasets": [
            {
                "name": f"yiddish_ass_{best_method}",
                "path": str(output_dir),
                "meta_file_train": f"yiddish_ass_{best_method}_train_data.txt",
                "meta_file_val": f"yiddish_ass_{best_method}_train_data.txt"  # Using same for now
            }
        ],
        
        # Audio settings
        "audio": {
            "sample_rate": sample_rate,
            "hop_length": 256 if sample_rate == 16000 else 512,
            "win_length": 800 if sample_rate == 16000 else 2048,
            "n_mel_channels": 80,
            "mel_fmin": 0,
            "mel_fmax": sample_rate // 2,
            "n_fft": 1024 if sample_rate == 16000 else 2048,
            "preemphasis": 0.97,
            "trim_db": 60
        },
        
        # Model architecture
        "model_args": {
            "n_speakers": 1,
            "use_speaker_embedding": False,
            "use_d_vector_file": False
        },
        
        # Training
        "batch_size": 16,
        "eval_batch_size": 8,
        "num_loader_workers": 4,
        "num_eval_loader_workers": 2,
        "epochs": 1000,
        "save_step": 500,
        "eval_step": 100,
        "print_step": 25,
        
        # Optimizer
        "optimizer": "RAdam",
        "optimizer_params": {
            "lr": 0.001,
            "weight_decay": 1e-6
        },
        
        # Learning rate scheduler
        "lr_scheduler": "NoamLR",
        "lr_scheduler_params": {
            "warmup_steps": 4000
        },
        
        # Text processing
        "characters": {
            "characters": characters_string,
            "punctuations": "!\"'(),-.:;? ",
            "phonemes": "",
            "is_unique": True,
            "is_sorted": False
        },
        
        "text_cleaner": "basic_cleaners",
        "enable_eos_bos_chars": False,
        "test_sentences": [
            "שלום עליכם",
            "גוט מארגן",
            "וואס מאכסטו?",
            segments[0]['text'] if segments else "טעסט"
        ],
        
        # Data filtering
        "min_text_len": 5,
        "max_text_len": int(avg_text_length * 3),  # 3x average length
        "min_audio_len": 0.5,
        "max_audio_len": 15.0,
        
        # Output
        "output_path": str(output_dir / "training_output"),
        
        # Mixed precision
        "mixed_precision": True,
        
        # Logging
        "dashboard_logger": "tensorboard",
        "tb_model_param_stats": True,
        
        # Distributed training
        "distributed_backend": "nccl",
        "distributed_url": "tcp://localhost:54321"
    }
    
    config_path = output_dir / f"fastspeech2_yiddish_ass_{best_method}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"  ⚙️  Config: {config_path}")
    print(f"  📊 Data stats: {len(segments)} segments, {total_duration:.1f}s total, {avg_text_length:.1f} chars avg")
    
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Convert .ass subtitle file to TTS training data")
    parser.add_argument("--ass_file", default="correct .ass/audio01_aegisub.ass", 
                       help="Path to .ass subtitle file")
    parser.add_argument("--audio_file", default="original_files/audio/audio1.wav",
                       help="Path to source audio file")
    parser.add_argument("--output_dir", default="ass_training_data",
                       help="Output directory for training data")
    parser.add_argument("--methods", nargs="+", 
                       choices=["precise", "padded", "crossfade", "silence_trimmed"],
                       default=["precise", "padded", "crossfade", "silence_trimmed"],
                       help="Audio extraction methods to use")
    
    args = parser.parse_args()
    
    print("🎬 Converting .ass file to TTS training data")
    print(f"📁 ASS file: {args.ass_file}")
    print(f"🎵 Audio file: {args.audio_file}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🔧 Methods: {', '.join(args.methods)}")
    
    # Check if files exist
    if not os.path.exists(args.ass_file):
        print(f"❌ Error: .ass file not found: {args.ass_file}")
        return
    
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        return
    
    # Create training data
    metadata = create_training_data(
        args.ass_file, 
        args.audio_file, 
        args.output_dir, 
        args.methods
    )
    
    print("\n🎉 Conversion complete!")
    print(f"\n🚀 To start training, run:")
    print(f"source tts_venv/bin/activate")
    print(f"tts --config_path {args.output_dir}/fastspeech2_yiddish_ass_{args.methods[0]}.json")

if __name__ == "__main__":
    main() 