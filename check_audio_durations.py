#!/usr/bin/env python3
import json
import torchaudio
import os
from statistics import mean

with open('tts_segments/segments_metadata.json', 'r') as f:
    metadata = json.load(f)

durations = []
for item in metadata[:20]:  # Check first 20 samples
    audio_path = item["audio_file"]  # Path is already complete in metadata
    if os.path.exists(audio_path):
        try:
            audio, sr = torchaudio.load(audio_path)
            duration = audio.shape[1] / sr
            durations.append(duration)
            print(f'{item["audio_file"]}: {duration:.2f}s')
        except Exception as e:
            print(f'Error loading {item["audio_file"]}: {e}')

if durations:
    print(f'\nSample count: {len(durations)}')
    print(f'Average duration: {mean(durations):.2f}s')
    print(f'Max duration: {max(durations):.2f}s')
    print(f'Min duration: {min(durations):.2f}s')
else:
    print('No audio files found') 