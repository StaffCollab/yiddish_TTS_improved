#!/usr/bin/env python3
"""
Safe Yiddish TTS Training - Resource-Friendly Version
Trains slowly to avoid overwhelming the system
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
import unicodedata
import re
import gc  # For garbage collection
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
from datetime import datetime


class YiddishTokenizer:
    """Proven Hebrew tokenizer"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
    def build_vocab_from_texts(self, texts):
        """Build vocabulary from Yiddish texts"""
        print("Building Hebrew vocabulary from full dataset...")
        
        all_chars = set()
        all_chars.update([self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN])
        
        for text in texts:
            normalized = self.normalize_text(text)
            all_chars.update(normalized)
        
        sorted_chars = sorted(list(all_chars))
        
        for idx, char in enumerate(sorted_chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        
        self.vocab_size = len(sorted_chars)
        
        print(f"✅ Vocabulary: {self.vocab_size} characters")
        print(f"Hebrew chars: {[c for c in sorted_chars if ord(c) >= 0x0590 and ord(c) <= 0x05FF][:10]}...")
        
    def normalize_text(self, text):
        """Normalize Hebrew text"""
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        normalized = self.normalize_text(text)
        sequence = [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) for char in normalized]
        return sequence


class LightweightEncoder(nn.Module):
    """Lightweight encoder to reduce memory usage"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):  # Reduced sizes
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Simpler convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            num_layers=1,  # Reduced layers
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
    def forward(self, text):
        embedded = self.embedding(text)
        conv_input = embedded.transpose(1, 2)
        conv_output = self.conv_layers(conv_input)
        conv_output = conv_output.transpose(1, 2)
        
        lstm_output, _ = self.lstm(conv_output)
        return lstm_output


class LightweightDecoder(nn.Module):
    """Lightweight decoder with attention"""
    
    def __init__(self, encoder_dim=256, decoder_dim=256, n_mels=80):  # Reduced sizes
        super().__init__()
        
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        
        # Simpler attention
        self.attention = nn.MultiheadAttention(decoder_dim, num_heads=4)  # Fewer heads
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            encoder_dim + n_mels, 
            decoder_dim, 
            num_layers=1,  # Single layer
            batch_first=True
        )
        
        # Output projections
        self.mel_projection = nn.Linear(decoder_dim, n_mels)
        self.stop_projection = nn.Linear(decoder_dim, 1)
        
        # Simple postnet
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_mels, n_mels, kernel_size=3, padding=1),
        )
        
    def forward(self, encoder_outputs, mel_targets=None):
        batch_size = encoder_outputs.size(0)
        max_decoder_steps = 500 if mel_targets is None else mel_targets.size(1)
        
        decoder_outputs = []
        stop_outputs = []
        
        # Initialize
        decoder_input = torch.zeros(batch_size, 1, self.n_mels).to(encoder_outputs.device)
        hidden = None
        
        for step in range(max_decoder_steps):
            # Attention
            attn_output, _ = self.attention(
                decoder_input.transpose(0, 1),
                encoder_outputs.transpose(0, 1),
                encoder_outputs.transpose(0, 1)
            )
            attn_output = attn_output.transpose(0, 1)
            
            # Concatenate with previous mel
            lstm_input = torch.cat([attn_output, decoder_input], dim=-1)
            
            # LSTM step
            lstm_output, hidden = self.decoder_lstm(lstm_input, hidden)
            
            # Projections
            mel_output = self.mel_projection(lstm_output)
            stop_output = self.stop_projection(lstm_output)
            
            decoder_outputs.append(mel_output)
            stop_outputs.append(stop_output)
            
            # Teacher forcing or previous output
            if mel_targets is not None and step < mel_targets.size(1) - 1:
                decoder_input = mel_targets[:, step:step+1, :]
            else:
                decoder_input = mel_output
            
            # Early stopping during inference
            if mel_targets is None and torch.sigmoid(stop_output).item() > 0.5:
                break
        
        mel_outputs = torch.cat(decoder_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        # Postnet
        mel_postnet = mel_outputs.transpose(1, 2)
        mel_postnet = self.postnet(mel_postnet)
        mel_postnet = mel_postnet.transpose(1, 2)
        mel_postnet = mel_outputs + mel_postnet
        
        return mel_outputs, mel_postnet, stop_outputs


class SafeYiddishTacotron(nn.Module):
    """Resource-friendly Tacotron model"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = LightweightEncoder(vocab_size)
        self.decoder = LightweightDecoder()
    
    def forward(self, text, mel_targets=None):
        encoder_outputs = self.encoder(text)
        mel_outputs, mel_postnet, stop_outputs = self.decoder(encoder_outputs, mel_targets)
        return mel_outputs, mel_postnet, stop_outputs


class SafeYiddishDataset(Dataset):
    """Memory-efficient dataset"""
    
    def __init__(self, metadata_path, tokenizer, max_samples=100, max_audio_length=5.0):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.tokenizer = tokenizer
        
        # Limit samples for memory
        self.metadata = self.metadata[:max_samples]
        print(f"Using {len(self.metadata)} samples (limited for memory safety)")
        
        # Filter by audio length
        self.filtered_metadata = []
        for item in self.metadata:
            try:
                audio_path = f"tts_segments/audio/{item['audio_file']}"
                if os.path.exists(audio_path):
                    audio, sr = torchaudio.load(audio_path)
                    duration = audio.shape[1] / sr
                    if duration <= max_audio_length:
                        self.filtered_metadata.append(item)
            except:
                continue
        
        self.metadata = self.filtered_metadata
        print(f"Filtered to {len(self.metadata)} samples (≤{max_audio_length}s)")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load and process audio
        audio_path = f"tts_segments/audio/{item['audio_file']}"
        audio, sr = torchaudio.load(audio_path)
        
        # Convert to mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=8000
        )
        
        mel_spec = mel_transform(audio)
        mel_spec = torch.log(mel_spec + 1e-9)  # Log mel
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # [time, mels]
        
        # Process text
        text = item['text']
        text_sequence = self.tokenizer.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence)
        
        return {
            'text': text_tensor,
            'mel_targets': mel_spec,
            'text_raw': text
        }


def collate_fn_safe(batch):
    """Safe collate function with memory management"""
    texts = [item['text'] for item in batch]
    mel_targets = [item['mel_targets'] for item in batch]
    text_raws = [item['text_raw'] for item in batch]
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    # Pad mel spectrograms
    max_mel_len = max(mel.size(0) for mel in mel_targets)
    mel_targets_padded = torch.zeros(len(mel_targets), max_mel_len, 80)
    
    for i, mel in enumerate(mel_targets):
        mel_len = mel.size(0)
        mel_targets_padded[i, :mel_len, :] = mel
    
    return {
        'text': texts_padded,
        'mel_targets': mel_targets_padded,
        'text_raw': text_raws
    }


def train_safe_yiddish():
    """Safe training with resource management"""
    print("=" * 70)
    print("🐌 SAFE YIDDISH TTS TRAINING")
    print("=" * 70)
    print("• Resource-friendly approach")
    print("• Small batches and incremental training")
    print("• Memory management and cleanup")
    print("• Frequent checkpointing")
    print("=" * 70)
    
    if not os.path.exists("tts_segments/segments_metadata.json"):
        print("❌ Error: Please ensure your data is in tts_segments/")
        return
    
    # Load metadata
    with open("tts_segments/segments_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    all_texts = [item['text'] for item in metadata]
    
    # Create tokenizer
    tokenizer = YiddishTokenizer()
    tokenizer.build_vocab_from_texts(all_texts)
    
    # Save tokenizer
    tokenizer_data = {
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'vocab_size': tokenizer.vocab_size
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer_file = f"yiddish_safe_tokenizer_{timestamp}.json"
    
    with open(tokenizer_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer saved: {tokenizer_file}")
    
    # Create safe dataset (fewer samples)
    dataset = SafeYiddishDataset(
        "tts_segments/segments_metadata.json",
        tokenizer,
        max_samples=50,  # Very limited for safety
        max_audio_length=3.0  # Shorter audio clips
    )
    
    # Tiny batch sizes
    train_loader = DataLoader(
        dataset, 
        batch_size=1,  # Single sample per batch
        shuffle=True, 
        collate_fn=collate_fn_safe,
        num_workers=0  # No multiprocessing
    )
    
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Create lightweight model
    model = SafeYiddishTacotron(vocab_size=tokenizer.vocab_size)
    
    device = torch.device('cpu')  # Force CPU for safety
    model.to(device)
    
    # Conservative optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)  # Lower learning rate
    
    # Loss functions
    mel_criterion = nn.MSELoss()
    stop_criterion = nn.BCEWithLogitsLoss()
    
    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Short training sessions
    num_epochs = 5  # Very few epochs per session
    best_loss = float('inf')
    
    print(f"🐌 Starting {num_epochs}-epoch training session...")
    print(f"⚠️  This will train slowly to protect your system")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Memory cleanup before each batch
            if batch_idx % 5 == 0:
                gc.collect()
            
            text = batch['text'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                mel_outputs, mel_postnet, stop_outputs = model(text, mel_targets)
                
                # Align dimensions carefully
                min_time = min(mel_outputs.size(1), mel_targets.size(1))
                mel_outputs_aligned = mel_outputs[:, :min_time, :]
                mel_postnet_aligned = mel_postnet[:, :min_time, :]
                mel_targets_aligned = mel_targets[:, :min_time, :]
                
                # Create stop targets
                stop_targets = torch.zeros(stop_outputs.size(0), min_time).to(device)
                stop_targets[:, -1] = 1.0
                
                # Calculate losses
                mel_loss = mel_criterion(mel_outputs_aligned, mel_targets_aligned)
                mel_postnet_loss = mel_criterion(mel_postnet_aligned, mel_targets_aligned)
                stop_loss = stop_criterion(stop_outputs[:, :min_time].squeeze(-1), stop_targets)
                
                # Combined loss
                loss = mel_loss + mel_postnet_loss + 0.5 * stop_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Conservative clipping
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Frequent progress updates
                if batch_idx % 2 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}")
                    print(f"    Loss: {loss.item():.4f}")
                    print(f"    Text: {batch['text_raw'][0][:30]}...")
                
                # Memory cleanup
                del mel_outputs, mel_postnet, stop_outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  ⚠️  Batch {batch_idx} failed: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start
            
            print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"   Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint every epoch (frequent saves)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer_data,
                'loss': avg_loss
            }
            
            checkpoint_file = f"yiddish_safe_checkpoint_epoch_{epoch+1}_{timestamp}.pth"
            torch.save(checkpoint, checkpoint_file)
            print(f"   💾 Checkpoint saved: {checkpoint_file}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_file = f"yiddish_safe_best_{timestamp}.pth"
                torch.save(checkpoint, best_file)
                print(f"   🏆 New best model: {best_file}")
        
        # Memory cleanup after epoch
        gc.collect()
        
        # Take a break between epochs
        print("   💤 Resting 3 seconds...")
        time.sleep(3)
    
    print("✅ Safe training session completed!")
    print(f"🔄 To continue training, run the script again")
    print(f"   (It will create new checkpoints and continue improving)")
    
    print("\n🎯 Next steps:")
    print("1. Run this script multiple times for gradual improvement")
    print("2. Use generate_yiddish_speech.py for zero-shot generation")
    print("3. Load checkpoints if you want to continue training")
    
    return tokenizer_file


if __name__ == "__main__":
    print("🐌 Safe Yiddish TTS Training")
    print("This version protects your system by using minimal resources")
    
    try:
        tokenizer_file = train_safe_yiddish()
        print(f"\n🎉 Training session successful!")
        print(f"📁 Tokenizer: {tokenizer_file}")
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Try reducing batch size or sample count further") 