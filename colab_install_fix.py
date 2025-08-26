#!/usr/bin/env python3
"""
Fixed installation script for Google Colab with Python 3.12
Run this in Colab cells for proper TTS installation
"""

# Cell 1: System dependencies and core packages
print("=" * 50)
print("STEP 1: Installing system dependencies")
print("=" * 50)
!apt-get update -qq
!apt-get install -y -qq libsndfile1 ffmpeg espeak-ng build-essential

print("\n" + "=" * 50)
print("STEP 2: Upgrading pip and core tools")
print("=" * 50)
!pip install --upgrade pip setuptools wheel

print("\n" + "=" * 50)
print("STEP 3: Installing PyTorch with CUDA")
print("=" * 50)
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

print("\n" + "=" * 50)
print("STEP 4: Installing audio libraries")
print("=" * 50)
!pip install numpy==1.24.3
!pip install librosa>=0.10.0
!pip install soundfile>=0.12.0
!pip install scipy>=1.11.0
!pip install pandas>=2.0.0

# Cell 2: TTS Installation with multiple fallbacks
print("\n" + "=" * 50)
print("STEP 5: Installing Coqui TTS")
print("=" * 50)

# Method 1: Try the standard installation
!pip install TTS==0.22.0

# Verify installation
import sys
import subprocess

def check_tts():
    try:
        import TTS
        from TTS.api import TTS as tts_api
        print("✅ TTS module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ TTS import failed: {e}")
        return False

if not check_tts():
    print("\n⚠️ Standard installation failed, trying GitHub install...")
    
    # Method 2: Uninstall and reinstall from GitHub
    !pip uninstall -y TTS
    !pip install git+https://github.com/coqui-ai/TTS.git
    
    if not check_tts():
        print("\n⚠️ GitHub install failed, trying alternative method...")
        
        # Method 3: Install with no dependencies then add them
        !pip uninstall -y TTS
        !pip install --no-deps TTS==0.22.0
        !pip install gruut>=2.2.3
        !pip install inflect>=5.6.0
        !pip install unidecode>=1.3.0
        !pip install pypinyin>=0.47.0
        !pip install mecab-python3>=1.0.5
        !pip install jamo>=0.4.1
        !pip install g2pkk>=0.1.1
        
        if not check_tts():
            print("\n❌ All installation methods failed!")
            print("Try restarting the runtime and running this alternative:")
            print("!pip install coqui-tts")
        else:
            print("✅ TTS installed with alternative method!")
    else:
        print("✅ TTS installed from GitHub!")
else:
    print("✅ TTS installed successfully!")

# Cell 3: Install remaining dependencies
print("\n" + "=" * 50)
print("STEP 6: Installing additional dependencies")
print("=" * 50)
!pip install matplotlib>=3.7.0
!pip install scikit-learn>=1.3.0
!pip install PyYAML>=6.0
!pip install tqdm>=4.64.0
!pip install tensorboard
!pip install psutil

# Optional: Whisper
print("\n" + "=" * 50)
print("STEP 7: Installing Whisper (optional)")
print("=" * 50)
!pip install -q openai-whisper

print("\n" + "=" * 50)
print("INSTALLATION COMPLETE!")
print("=" * 50)

# Final verification
print("\nFinal package verification:")
packages = ['torch', 'torchaudio', 'numpy', 'librosa', 'soundfile', 'TTS']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {pkg}: {version}")
    except ImportError:
        print(f"❌ {pkg}: NOT INSTALLED")