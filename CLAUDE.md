# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Yiddish Text-to-Speech (TTS) system, one of the first functional Yiddish TTS implementations. The project uses Hebrew script characters for Yiddish text and employs various TTS architectures including Tacotron2 and XTTS.

## Key Commands

### Environment Setup
```bash
# Activate virtual environment (Windows)
tts_venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source tts_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Yiddish Speech
```bash
# Generate speech from text (immediate, no training required)
python generate_yiddish_speech.py "שבת שלום"

# Test with sample phrases
python generate_yiddish_speech.py
```

### Training Models

```bash
# Fast Tacotron2 training (RECOMMENDED for initial training)
python train_yiddish_tacotron2_fast.py

# XTTS training (highest quality, advanced features)
python train_xtts_yiddish_perfect.py

# Full Tacotron2 with evaluation
python train_yiddish_tacotron2_working.py

# Safe training (resource-friendly)
python train_full_yiddish_safe.py
```

### Data Preparation
```bash
# Prepare perfect aligned segments
python prepare_perfect_data.py

# Create perfect audio-text alignments
python batch_perfect_mapper.py
```

## Architecture & Key Components

### Core Training Scripts
- **train_yiddish_tacotron2_fast.py**: Optimized Tacotron2 training without evaluation for speed
- **train_xtts_yiddish_perfect.py**: State-of-the-art XTTS architecture for production quality
- **train_yiddish_tacotron2_working.py**: Full Tacotron2 with evaluation pipeline
- **train_yiddish_tacotron2_natural.py**: Natural speech variant with specific config

### Data Processing Pipeline
1. **Audio Alignment**: Multiple alignment scripts (whisper_alignment.py, forced_alignment_solution.py) to create perfect audio-text pairs
2. **Text Processing**: YiddishTextProcessor class handles Hebrew script normalization and character tokenization
3. **Segmentation**: Various segmentation approaches (smart_audio_segmentation.py, tts_optimal_segmentation.py)

### Key Directories
- **perfect_mapped_segments/**: High-quality aligned audio-text dataset with metadata
- **yiddish_tacotron2_training/**: Active training runs and model checkpoints
- **tts_segments/**: Legacy dataset (272 audio-text pairs)
- **yiddish_tts_output/**: Generated speech outputs

### Configuration
- **final_yiddish_config.json**: Main Tacotron2 configuration with Hebrew character support
- **yiddish_custom_config.json**: Alternative configurations
- Character set: 54+ unique characters including Hebrew letters (אבגדהוזחטיךכלםמןנסעףפץצקרשת)
- Audio: 16kHz sample rate (optimized), 22kHz (legacy)

## Hebrew Script Handling

The system properly handles right-to-left (RTL) Hebrew script:
- Text normalization via YiddishTextProcessor class
- Unicode NFD normalization for consistency
- Special handling for Hebrew punctuation (geresh, gershayim)
- Character tokenization includes all Hebrew letters plus diacritics

## Testing & Validation

```bash
# Test trained model
python test_trained_model.py

# Test Hebrew directionality
python test_hebrew_directionality.py

# Verify audio-text alignment
python verify_audio_text_alignment.py
```

## Important Notes

- Always activate the virtual environment before running scripts
- The project uses Coqui TTS library (>=0.22.0) with PyTorch
- Hebrew language ("he") is used as the closest approximation for Yiddish in multilingual models
- Training outputs are saved to yiddish_tacotron2_training/ directory
- Perfect mapped segments provide higher quality than original tts_segments dataset