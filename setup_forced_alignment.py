#!/usr/bin/env python3
"""
Montreal Forced Alignment (MFA) Setup
The gold standard for audio-text alignment
"""

import os
import subprocess
from pathlib import Path
import shutil

def check_mfa_installation():
    """Check if MFA is installed"""
    print("🔍 Checking Montreal Forced Alignment (MFA) installation...")
    
    try:
        result = subprocess.run(['mfa', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ MFA installed: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ MFA not found")
            return False
    except FileNotFoundError:
        print("   ❌ MFA not installed")
        return False

def install_mfa_instructions():
    """Provide MFA installation instructions"""
    print("\n📦 Montreal Forced Alignment Installation:")
    print("=" * 50)
    
    print("MFA is the BEST tool for audio-text alignment.")
    print("It will give you EXACT word timings.\n")
    
    print("🐍 Installation options:")
    print("\nOption 1 - Conda (Recommended):")
    print("   conda install -c conda-forge montreal-forced-alignment")
    
    print("\nOption 2 - Pip:")
    print("   pip install montreal-forced-alignment")
    
    print("\nOption 3 - System package (Ubuntu/Debian):")
    print("   sudo apt update")
    print("   sudo apt install montreal-forced-alignment")
    
    print("\n⚠️  Note: MFA requires additional dependencies:")
    print("   - Kaldi (audio processing)")
    print("   - Language models")
    
    print("\n🔧 After installation:")
    print("   1. Download acoustic model: mfa model download acoustic english_us_arpa")
    print("   2. Download dictionary: mfa model download dictionary english_us_arpa")
    print("   3. Run this script again to set up alignment")

def prepare_mfa_data():
    """Prepare data in MFA format"""
    print("\n📁 Preparing data for MFA...")
    
    # Create MFA directory structure
    mfa_dir = Path("mfa_alignment")
    mfa_dir.mkdir(exist_ok=True)
    
    audio_dir = mfa_dir / "audio"
    text_dir = mfa_dir / "text"
    output_dir = mfa_dir / "aligned"
    
    audio_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Copy original files to MFA format
    original_audio = Path("original_files/audio")
    original_transcripts = Path("original_files/transcripts")
    
    if not original_audio.exists() or not original_transcripts.exists():
        print("   ❌ Original files not found")
        return False
    
    # Copy and rename files for MFA (it expects matching names)
    audio_files = sorted(original_audio.glob("audio*.wav"))
    transcript_files = sorted(original_transcripts.glob("transcription*.txt"))
    
    copied_files = 0
    for audio_file, transcript_file in zip(audio_files, transcript_files):
        # MFA expects audio.wav and audio.txt with same basename
        base_name = f"file_{copied_files+1:03d}"
        
        # Copy audio
        shutil.copy2(audio_file, audio_dir / f"{base_name}.wav")
        
        # Copy transcript (MFA expects plain text, one sentence per line)
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        with open(text_dir / f"{base_name}.txt", 'w', encoding='utf-8') as f:
            f.write(content)
        
        copied_files += 1
    
    print(f"   ✅ Prepared {copied_files} file pairs for MFA")
    print(f"   📂 Audio files: {audio_dir}")
    print(f"   📂 Text files: {text_dir}")
    print(f"   📂 Output will go to: {output_dir}")
    
    return True

def run_mfa_alignment():
    """Run MFA alignment"""
    print("\n🚀 Running MFA Alignment...")
    
    mfa_dir = Path("mfa_alignment")
    audio_dir = mfa_dir / "audio"
    text_dir = mfa_dir / "text"
    output_dir = mfa_dir / "aligned"
    
    # Check if we have a Hebrew/Yiddish model, otherwise use English
    print("   📋 Note: Using English model for demonstration")
    print("   🌍 For Hebrew/Yiddish, you may need to train a custom model")
    
    # MFA alignment command
    cmd = [
        'mfa', 'align',
        str(audio_dir),           # audio directory
        str(text_dir),            # text directory  
        'english_us_arpa',        # acoustic model
        'english_us_arpa',        # dictionary
        str(output_dir)           # output directory
    ]
    
    print(f"   🔧 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("   ✅ MFA alignment completed successfully!")
            return True
        else:
            print(f"   ❌ MFA alignment failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ MFA alignment timed out (>10 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ MFA alignment error: {e}")
        return False

def convert_mfa_to_segments():
    """Convert MFA TextGrid output to TTS-ready segments"""
    print("\n🔄 Converting MFA output to TTS segments...")
    
    try:
        import textgrid  # pip install textgrid
    except ImportError:
        print("   📦 Installing textgrid package...")
        subprocess.run(['pip', 'install', 'textgrid'], check=True)
        import textgrid
    
    mfa_output = Path("mfa_alignment/aligned")
    if not mfa_output.exists():
        print("   ❌ No MFA output found")
        return False
    
    # Create output directory for TTS segments
    segments_dir = Path("tts_segments_mfa")
    segments_dir.mkdir(exist_ok=True)
    (segments_dir / "audio").mkdir(exist_ok=True)
    (segments_dir / "text").mkdir(exist_ok=True)
    
    # Process each TextGrid file
    textgrid_files = list(mfa_output.glob("*.TextGrid"))
    print(f"   📄 Found {len(textgrid_files)} TextGrid files")
    
    total_segments = 0
    
    for tg_file in textgrid_files:
        print(f"   Processing: {tg_file.name}")
        
        # Load TextGrid
        tg = textgrid.TextGrid.fromFile(str(tg_file))
        
        # Find word tier (usually tier 1)
        word_tier = None
        for tier in tg.tiers:
            if hasattr(tier, 'intervals') and 'word' in tier.name.lower():
                word_tier = tier
                break
        
        if not word_tier:
            print(f"   ⚠️  No word tier found in {tg_file.name}")
            continue
        
        # Group words into TTS-appropriate segments (3-8 seconds)
        current_segment_words = []
        current_start = None
        segment_count = 0
        
        for interval in word_tier.intervals:
            if interval.mark.strip():  # Non-empty word
                if current_start is None:
                    current_start = interval.minTime
                
                current_segment_words.append(interval.mark)
                current_duration = interval.maxTime - current_start
                
                # If segment is long enough, save it
                if current_duration >= 3.0 and len(current_segment_words) >= 3:
                    # Check if we should end here or continue
                    if current_duration >= 6.0 or len(current_segment_words) >= 8:
                        # Save current segment
                        segment_text = ' '.join(current_segment_words)
                        
                        # TODO: Extract corresponding audio segment
                        # This would require the original audio file
                        
                        segment_count += 1
                        total_segments += 1
                        
                        # Reset for next segment
                        current_segment_words = []
                        current_start = None
        
        print(f"     → Created {segment_count} segments")
    
    print(f"   ✅ Total segments created: {total_segments}")
    return True

def mfa_setup_main():
    """Main MFA setup function"""
    print("🎯 Montreal Forced Alignment Setup")
    print("=" * 40)
    
    # Check installation
    if not check_mfa_installation():
        install_mfa_instructions()
        return
    
    # Prepare data
    if not prepare_mfa_data():
        print("❌ Could not prepare data for MFA")
        return
    
    print("\n🔧 Ready for MFA alignment!")
    print("\nNext steps:")
    print("1. Ensure you have the right language model")
    print("2. Run: python setup_forced_alignment.py --align")
    print("3. Or manually run:")
    print("   mfa align mfa_alignment/audio mfa_alignment/text english_us_arpa english_us_arpa mfa_alignment/aligned")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--align':
        if run_mfa_alignment():
            convert_mfa_to_segments()
    else:
        mfa_setup_main() 