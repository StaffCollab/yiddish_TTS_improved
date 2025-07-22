#!/usr/bin/env python3
"""
Generate Yiddish Speech from Trained Conservative Checkpoint
Use your trained conservative model to synthesize Yiddish speech
"""

import os
import sys
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
import json
import unicodedata
import re

# EXACT model architecture from train_full_yiddish_conservative.py
class YiddishTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
    
    def build_vocab_from_texts(self, texts):
        chars = set()
        for text in texts:
            for char in text:
                if len(char) == 1:  # Only single characters
                    chars.add(char)
        
        # Sort for consistency
        chars = sorted(list(chars))
        
        # Add special tokens
        special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        all_chars = special_tokens + chars
        
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        
        print(f"   Vocabulary: {self.vocab_size} characters")
        print(f"   Characters: {''.join(chars)}")
    
    def encode(self, text, max_length=None):
        indices = [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) for char in text]
        if max_length:
            if len(indices) < max_length:
                indices.extend([self.char_to_idx[self.PAD_TOKEN]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        chars = [self.idx_to_char.get(idx, '') for idx in indices]
        return ''.join(chars).replace(self.PAD_TOKEN, '').replace(self.SOS_TOKEN, '').replace(self.EOS_TOKEN, '')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, mel_dim=80, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(mel_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Attention
        self.attention = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Output projection
        self.mel_projection = nn.Linear(hidden_dim * 2, mel_dim)
        self.stop_projection = nn.Linear(hidden_dim * 2, 1)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, encoder_outputs, max_length=1000):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize
        mel_outputs = []
        stop_outputs = []
        
        # Initial mel frame (zeros)
        prev_mel = torch.zeros(batch_size, 1, 80, device=device)
        
        # Hidden states
        h1 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c1 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        h2 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c2 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        h3 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c3 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        
        for i in range(max_length):
            # Attention
            query = h3.transpose(0, 1)  # (batch, 1, hidden_dim)
            attention_scores = torch.bmm(
                torch.tanh(self.attention(torch.cat([
                    query.expand(-1, encoder_outputs.size(1), -1),
                    encoder_outputs
                ], dim=2))),
                encoder_outputs.transpose(1, 2)
            ).squeeze(2)
            
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(1)
            context = torch.bmm(attention_weights, encoder_outputs)
            
            # Decoder input
            decoder_input = torch.cat([prev_mel, context], dim=2)
            
            # LSTM layers
            lstm1_out, (h1, c1) = self.lstm1(decoder_input, (h1, c1))
            lstm1_out = self.dropout(lstm1_out)
            
            lstm2_out, (h2, c2) = self.lstm2(lstm1_out, (h2, c2))
            lstm2_out = self.dropout(lstm2_out)
            
            lstm3_out, (h3, c3) = self.lstm3(lstm2_out, (h3, c3))
            
            # Output projections
            decoder_output = torch.cat([lstm3_out, context], dim=2)
            mel_output = self.mel_projection(decoder_output)
            stop_output = torch.sigmoid(self.stop_projection(decoder_output))
            
            mel_outputs.append(mel_output)
            stop_outputs.append(stop_output)
            
            # Use current output as next input
            prev_mel = mel_output
            
            # Stop condition
            if stop_output.squeeze().item() > 0.5:
                break
        
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs

class ConservativeYiddishTacotron(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=256, mel_dim=80):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, mel_dim)
    
    def forward(self, text_input, mel_target=None):
        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(text_input)
        
        if mel_target is not None:
            # Training mode - teacher forcing
            decoder_output, stop_output = self.decoder(encoder_outputs, mel_target.size(1))
        else:
            # Inference mode
            decoder_output, stop_output = self.decoder(encoder_outputs)
        
        return decoder_output, stop_output

def load_trained_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    print(f"🔄 Loading trained model from: {Path(checkpoint_path).name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("   ✅ Checkpoint loaded")
    print(f"   📊 Training info:")
    if 'epoch' in checkpoint:
        print(f"      Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"      Loss: {checkpoint['loss']:.4f}")
    
    # Create tokenizer (need to rebuild vocab)
    tokenizer = YiddishTokenizer()
    
    # Load tokenizer if saved with checkpoint
    if 'tokenizer' in checkpoint:
        tokenizer.char_to_idx = checkpoint['tokenizer']['char_to_idx']
        tokenizer.idx_to_char = checkpoint['tokenizer']['idx_to_char'] 
        tokenizer.vocab_size = checkpoint['tokenizer']['vocab_size']
        print(f"   ✅ Tokenizer loaded: {tokenizer.vocab_size} characters")
    else:
        # Rebuild tokenizer from metadata
        print("   🔄 Rebuilding tokenizer from data...")
        with open('tts_segments/segments_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        all_texts = [item['text'] for item in metadata]
        tokenizer.build_vocab_from_texts(all_texts)
    
    # Create model
    model = ConservativeYiddishTacotron(vocab_size=tokenizer.vocab_size)
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Model weights loaded")
    else:
        print("   ❌ No model weights found in checkpoint")
        return None, None
    
    model.eval()
    return model, tokenizer

def generate_yiddish_speech(model, tokenizer, text, output_file="generated_yiddish.wav"):
    """Generate speech from text using trained model"""
    print(f"\n🎤 Generating Yiddish Speech")
    print(f"   Text: {text}")
    print(f"   Output: {output_file}")
    
    # Encode text
    text_tensor = tokenizer.encode(text).unsqueeze(0)  # Add batch dimension
    print(f"   Encoded length: {text_tensor.size(1)} characters")
    
    # Generate mel spectrogram
    with torch.no_grad():
        mel_output, stop_output = model(text_tensor)
    
    print(f"   Generated mel: {mel_output.shape}")
    
    # Convert mel to audio (simplified approach)
    # Note: A real vocoder (like WaveGlow) would be better
    mel_np = mel_output.squeeze().cpu().numpy()
    
    # Simple mel-to-audio conversion (basic approach)
    # This is a simplified version - real systems use neural vocoders
    audio = convert_mel_to_audio_simple(mel_np)
    
    # Save audio
    sample_rate = 22050
    torchaudio.save(output_file, torch.tensor(audio).unsqueeze(0), sample_rate)
    
    print(f"   ✅ Audio saved: {output_file}")
    print(f"   Duration: {len(audio)/sample_rate:.1f} seconds")
    
    return output_file

def convert_mel_to_audio_simple(mel_spec):
    """Simple mel spectrogram to audio conversion"""
    # This is a basic approach - real TTS uses neural vocoders
    
    # Convert mel spectrogram back to linear spectrogram (simplified)
    linear_spec = np.exp(mel_spec)  # Rough approximation
    
    # Use Griffin-Lim algorithm for phase reconstruction
    import librosa
    
    # Pad to reasonable size
    if linear_spec.shape[1] < 100:
        linear_spec = np.pad(linear_spec, ((0, 0), (0, 100 - linear_spec.shape[1])), mode='constant')
    
    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(linear_spec, n_iter=32, hop_length=256, n_fft=1024)
    
    return audio

def main():
    """Main generation function"""
    print("🎯 Yiddish Speech Generation from Trained Checkpoint")
    print("=" * 60)
    
    # Find latest checkpoint (epoch 5)
    checkpoint_path = "yiddish_conservative_checkpoint_epoch_5_20250721_221138.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for file in Path(".").glob("yiddish_conservative_checkpoint_*.pth"):
            print(f"   {file.name}")
        return
    
    # Load model
    model, tokenizer = load_trained_model(checkpoint_path)
    if model is None:
        print("❌ Failed to load model")
        return
    
    # Test text
    if len(sys.argv) > 1:
        test_text = sys.argv[1]
    else:
        # Default Yiddish text
        test_text = "שבת שלום, ווי גייט עס?"
    
    # Generate speech
    output_file = f"yiddish_generated_{len(test_text)}_chars.wav"
    generate_yiddish_speech(model, tokenizer, test_text, output_file)
    
    print(f"\n🎉 Success! Generated Yiddish speech:")
    print(f"   📁 File: {output_file}")
    print(f"   🎵 Play with: vlc {output_file}")
    print(f"   📝 Text: {test_text}")

if __name__ == "__main__":
    main() 