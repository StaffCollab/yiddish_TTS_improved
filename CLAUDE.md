# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Yiddish Text-to-Speech (TTS) project - one of the first functional Yiddish TTS systems. The project handles Hebrew script characters and uses state-of-the-art TTS architectures including XTTS v2 and Tacotron2 to generate Yiddish speech synthesis.

## Environment Setup

### Python Virtual Environment
```bash
# Activate the virtual environment (Windows)
.\tts_venv\Scripts\activate
# Or for local environment
.\local_tts_venv\Scripts\activate

# Linux/Mac alternative
source tts_venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies: torch>=2.1, TTS>=0.22.0, librosa>=0.10.0, whisper packages for alignment

## Core Development Commands

### Speech Generation (Ready to Use)
```bash
# Generate Yiddish speech immediately using XTTS v2
python generate_yiddish_speech.py "שבת שלום, ווי גייט עס?"

# Test with sample phrases
python generate_yiddish_speech.py
```

### Training Commands (Production Ready)

**Recommended Fast Training:**
```bash
python train_yiddish_tacotron2_working.py
```

**XTTS Training (Highest Quality):**
```bash
# Look for train_xtts_yiddish_perfect.py or similar XTTS training script
python train_xtts_yiddish_perfect.py
```

**Legacy Training Options:**
```bash
python train_full_yiddish_safe.py      # Resource-friendly
python train_full_yiddish.py           # Full training approach
python train_yiddish_simple.py         # Multilingual fine-tuning
```

### Data Preparation
```bash
# Prepare high-quality aligned dataset
python prepare_perfect_data.py

# Create perfect audio-text alignments
python batch_perfect_mapper.py

# Various alignment and segmentation tools
python whisper_alignment_fixed.py      # Fixed Whisper alignment
python smart_audio_segmentation.py     # Smart segmentation
python forced_alignment_solution.py    # Forced alignment
```

### Testing and Validation
```bash
# Test trained models
python test_trained_model.py
python test_my_model.py

# Validate alignments
python verify_audio_text_alignment.py
python verify_audio_text_pairing.py

# Check audio quality
python check_audio_durations.py
python quick_segment_checker.py
```

## Architecture and Data Structure

### Key Directories
- **`perfect_mapped_segments/`** - High-quality aligned audio-text pairs (production dataset)
- **`tts_segments/`** - Original 272 audio-text pairs (legacy)
- **`yiddish_tacotron2_training/`** - Active training runs and model checkpoints
- **`tts_venv/` or `local_tts_venv/`** - Python virtual environments

### Training Data Flow
1. **Raw Data** → Audio files + transcripts in `original_files/`
2. **Alignment** → Whisper-based alignment creates precise audio-text segments
3. **Perfect Mapping** → `batch_perfect_mapper.py` creates `perfect_mapped_segments/`
4. **Training** → Scripts use perfect segments to train TTS models
5. **Generation** → `generate_yiddish_speech.py` uses trained or pre-trained models

### Text Processing Architecture
The project includes specialized Yiddish text processors (`YiddishTextProcessor` classes) that:
- Handle Hebrew script characters (Unicode ranges 0x0590-0x05FF, 0xFB1D-0xFB4F)
- Process Right-to-Left (RTL) text properly
- Normalize Unicode text for TTS training
- Support 54+ unique Yiddish characters

### Model Architectures
- **XTTS v2**: State-of-the-art multilingual TTS with voice cloning
- **Tacotron2**: Classic neural TTS architecture optimized for Yiddish
- **Custom Models**: Hebrew script-aware modifications

## Configuration Files

- **`final_yiddish_config.json`** - Primary configuration for training
- **`yiddish_config.json`** - Legacy dataset configuration  
- **`yiddish_custom_config.json`** - Custom training parameters
- **`yiddish_config_FIXED.json`** - Fixed configuration issues

## Important Implementation Notes

### Hebrew Script Handling
- Text is processed with Hebrew character set awareness
- RTL text direction is handled properly
- Unicode normalization (NFD) is applied consistently
- Hebrew presentation forms are supported

### Audio Processing
- Primary sample rate: 16kHz (optimized for training speed)
- Legacy support: 22kHz
- Audio format: WAV files
- Whisper-based alignment for precise timing

### Training Considerations
- Use `perfect_mapped_segments/` for best quality training data
- Hebrew language code ("he") is used as closest match to Yiddish in multilingual models
- Character-level tokenization works better than word-level for Hebrew script

## Common Issues and Solutions

### Environment Issues
- Ensure correct virtual environment activation
- TTS library version 0.22.0+ required for proper Hebrew support
- torch and torchaudio versions must be compatible

### Training Issues
- If alignment fails, try different Whisper models or manual alignment tools
- For memory issues, use `train_full_yiddish_safe.py` 
- Check audio file integrity with `check_audio_durations.py`

### Text Processing Issues
- Ensure Unicode normalization is applied consistently
- Use RTL-aware text processing for proper character order
- Validate character sets match training expectations

## Docker Support

The project includes Docker configuration:
```bash
# Build Docker image (check Dockerfile for details)
docker build -t yiddish-tts .

# Run training in Docker
# See run_docker.ps1 for Windows PowerShell execution
```