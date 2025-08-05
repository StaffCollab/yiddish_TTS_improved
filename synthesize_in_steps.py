import torch
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.api import TTS
import os

# --- Configuration ---
# Use the verified paths to your best model and config
MODEL_PATH = "output/run-August-04-2025_07+12PM-ac742f0/best_model_7620.pth"
CONFIG_PATH = "output/run-August-04-2025_07+12PM-ac742f0/config.json"
VOCODER_NAME = "vocoder_models/universal/libri-tts/wavegrad"
TEXT_TO_SYNTHESIZE = "שבת שלום און א גוטן טאג"

# Intermediate and final file paths
SPECTROGRAM_PATH = "intermediate_spectrogram.pt"
OUTPUT_WAV_PATH = "test_synthesis_twostep.wav"

# --- Step 1: Generate Spectrogram from your Tacotron 2 Model ---
print("--- Step 1: Loading your trained Tacotron 2 model ---")

# Load config from the training run
config = Tacotron2Config()
config.load_json(CONFIG_PATH)

# Load your trained model
model = Tacotron2.init_from_config(config)
model.load_checkpoint(config, MODEL_PATH, eval=True, use_cuda=False)
model.to("cpu") # Ensure it runs on the CPU

print("--- Generating spectrogram... ---")
# Preprocess the text to get token IDs
text_inputs, _ = model.tokenizer.text_to_ids(TEXT_TO_SYNTHESIZE)

# Run the model's inference function
with torch.no_grad():
    # We use the postnet_outputs as it's the refined spectrogram
    _, postnet_outputs, _, _ = model.inference(torch.LongTensor(text_inputs).unsqueeze(0).to("cpu"))

# Save the generated spectrogram to a file
torch.save(postnet_outputs, SPECTROGRAM_PATH)
print(f"✅ Spectrogram saved to: {SPECTROGRAM_PATH}")

# Clear the Tacotron 2 model from memory
del model
print("--- Your Tacotron 2 model has been unloaded from memory. ---")


# --- Step 2: Convert Spectrogram to Audio using the Vocoder ---
print("\n--- Step 2: Loading WaveGrad vocoder ---")

# Load the pre-trained WaveGrad vocoder model. This will auto-download if needed.
# We explicitly set gpu=False to ensure it uses the CPU.
api = TTS(model_name=VOCODER_NAME, progress_bar=True, gpu=False)

# Load the spectrogram tensor from the file we saved in Step 1
spectrogram_tensor = torch.load(SPECTROGRAM_PATH)

print("--- Converting spectrogram to audio... ---")
# Use the vocoder to turn the spectrogram into a waveform
wav_output = api.vocoder.inference(spectrogram_tensor.to("cpu"))

# Save the final high-quality audio
api.synthesizer.save_wav(wav_output, OUTPUT_WAV_PATH)

print(f"\n🎉 Success! Final audio saved to: {OUTPUT_WAV_PATH}") 