#!/usr/bin/env python3
"""
Full Yiddish TTS Training - Complete Dataset
Train on all 272 samples with proven architecture and WaveGlow vocoder
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
        
        hebrew_chars = [c for c in sorted_chars if len(c) == 1 and ord(c) >= 0x0590 and ord(c) <= 0x05FF]
        print(f"✅ Full vocabulary: {self.vocab_size} chars, Hebrew: {len(hebrew_chars)}")
        
        return self.char_to_idx, self.idx_to_char
    
    def normalize_text(self, text):
        """Normalize Yiddish text"""
        if not text:
            return ""
        
        text = unicodedata.normalize('NFD', text)
        normalized = ""
        
        for char in text:
            if 0x0590 <= ord(char) <= 0x05FF:  # Hebrew
                normalized += char
            elif char in " .,!?;:-()[]{}\"'\n":  # Punctuation
                normalized += char
            elif char.isdigit():  # Numbers
                normalized += char
        
        return re.sub(r'\s+', ' ', normalized.strip())
    
    def text_to_sequence(self, text):
        """Convert text to sequence"""
        normalized = self.normalize_text(text)
        sequence = [self.char_to_idx.get(self.START_TOKEN, 0)]
        
        for char in normalized:
            idx = self.char_to_idx.get(char, self.char_to_idx.get(self.UNK_TOKEN, 0))
            sequence.append(idx)
        
        sequence.append(self.char_to_idx.get(self.END_TOKEN, 0))
        return sequence


class FullYiddishDataset(Dataset):
    """Dataset using the entire Yiddish corpus"""
    
    def __init__(self, metadata_file, tokenizer, sample_rate=22050, max_audio_length=10.0):
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_audio_samples = int(max_audio_length * sample_rate)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Filter for quality but keep most samples
        filtered_metadata = []
        for item in metadata:
            text = self.tokenizer.normalize_text(item['text'])
            duration = item.get('duration', 0)
            
            # Keep samples with reasonable text length and duration
            if (len(text) >= 3 and len(text) <= 200 and 
                duration > 0.5 and duration <= max_audio_length):
                filtered_metadata.append(item)
        
        self.metadata = filtered_metadata
        print(f"Using {len(self.metadata)}/{len(metadata)} samples (filtered for quality)")
        
        # Calculate statistics
        text_lengths = [len(self.tokenizer.normalize_text(item['text'])) for item in self.metadata]
        durations = [item['duration'] for item in self.metadata]
        
        print(f"Text length: {min(text_lengths)}-{max(text_lengths)} chars (avg: {sum(text_lengths)/len(text_lengths):.1f})")
        print(f"Audio duration: {min(durations):.1f}-{max(durations):.1f}s (avg: {sum(durations)/len(durations):.1f}s)")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio
        audio_path = Path(item['audio_file'])
        audio, sr = torchaudio.load(audio_path)
        
        # Proper audio preprocessing
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Limit audio length
        if audio.shape[1] > self.max_audio_samples:
            audio = audio[:, :self.max_audio_samples]
        
        # Normalize audio
        if torch.max(torch.abs(audio)) > 0:
            audio = audio / torch.max(torch.abs(audio))
        
        # Tokenize text
        text_sequence = self.tokenizer.text_to_sequence(item['text'])
        
        return {
            'audio': audio.squeeze(),
            'text': torch.LongTensor(text_sequence),
            'text_raw': item['text'],
            'duration': item['duration']
        }


class EnhancedEncoder(nn.Module):
    """Enhanced encoder for better text understanding"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers for character-level features
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            num_layers=3,
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, text):
        # Embedding
        embedded = self.embedding(text)  # [B, T, embedding_dim]
        
        # Convolutional processing
        conv_input = embedded.transpose(1, 2)  # [B, embedding_dim, T]
        conv_output = self.conv_layers(conv_input)
        conv_output = conv_output.transpose(1, 2)  # [B, T, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(conv_output)
        
        return self.dropout(lstm_out)


class AttentionDecoder(nn.Module):
    """Enhanced attention decoder"""
    
    def __init__(self, encoder_dim=512, decoder_dim=1024, n_mels=80):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_mels = n_mels
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            encoder_dim, 
            num_heads=8, 
            batch_first=True,
            dropout=0.1
        )
        
        # Decoder layers
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        self.decoder_rnn = nn.LSTM(
            encoder_dim + 256, 
            decoder_dim, 
            batch_first=True,
            num_layers=2,
            dropout=0.1
        )
        
        # Output projections
        self.mel_projection = nn.Linear(decoder_dim, n_mels)
        self.stop_projection = nn.Linear(decoder_dim, 1)
        
        # PostNet for refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='tanh')
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, encoder_outputs, mel_targets=None):
        batch_size, max_time, _ = encoder_outputs.shape
        
        if mel_targets is not None:
            max_decoder_steps = mel_targets.size(1)
        else:
            max_decoder_steps = max_time * 5  # Allow longer generation
        
        # Initialize
        decoder_input = torch.zeros(batch_size, 1, self.n_mels).to(encoder_outputs.device)
        hidden = None
        
        mel_outputs = []
        stop_outputs = []
        
        for step in range(max_decoder_steps):
            # PreNet processing
            prenet_output = self.prenet(decoder_input)
            
            # Attention
            attended, _ = self.attention(
                encoder_outputs,  # query
                encoder_outputs,  # key  
                encoder_outputs   # value
            )
            
            # Take weighted average based on step
            step_weight = min(step / max_decoder_steps, 1.0)
            context = attended.mean(dim=1, keepdim=True) * (1 - step_weight) + attended[:, step % max_time:step % max_time + 1] * step_weight
            
            # Combine context with prenet output
            decoder_input_combined = torch.cat([context, prenet_output], dim=-1)
            
            # Decoder step
            output, hidden = self.decoder_rnn(decoder_input_combined, hidden)
            
            # Generate outputs
            mel_output = self.mel_projection(output)
            stop_output = self.stop_projection(output)
            
            mel_outputs.append(mel_output)
            stop_outputs.append(stop_output)
            
            # Prepare next input
            if mel_targets is not None and step < max_decoder_steps - 1:
                # Teacher forcing
                decoder_input = mel_targets[:, step:step+1, :]
            else:
                # Use prediction
                decoder_input = mel_output
                
                # Stop condition for inference
                if mel_targets is None and torch.sigmoid(stop_output).item() > 0.8:
                    break
        
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        # PostNet refinement
        mel_outputs_postnet = mel_outputs.transpose(1, 2)  # [B, n_mels, T]
        mel_residual = self.postnet(mel_outputs_postnet)
        mel_outputs_postnet = mel_outputs + mel_residual.transpose(1, 2)
        
        return mel_outputs, mel_outputs_postnet, stop_outputs


class FullYiddishTacotron(nn.Module):
    """Enhanced Tacotron for full dataset training"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EnhancedEncoder(vocab_size)
        self.decoder = AttentionDecoder()
    
    def forward(self, text, mel_targets=None):
        encoder_outputs = self.encoder(text)
        mel_outputs, mel_postnet, stop_outputs = self.decoder(encoder_outputs, mel_targets)
        return mel_outputs, mel_postnet, stop_outputs


def create_enhanced_mel(audio, sample_rate=22050, n_mels=80, hop_length=256):
    """Create enhanced mel spectrograms with full dynamic range"""
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=hop_length,
        win_length=1024,
        power=2.0,
        f_min=0.0,
        f_max=sample_rate // 2,
        pad_mode='reflect'
    )
    
    mel_spec = mel_transform(audio)
    
    # Enhanced log conversion to capture full dynamic range
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-8))
    
    # Normalize to reasonable range while preserving dynamics
    mel_spec = mel_spec.transpose(-1, -2)
    
    return mel_spec


def collate_fn_full(batch):
    """Enhanced collate function for full dataset"""
    
    # Sort by text length for better batching
    batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
    
    # Pad text sequences
    text_sequences = [item['text'] for item in batch]
    text_lengths = [len(seq) for seq in text_sequences]
    text_padded = pad_sequence(text_sequences, batch_first=True, padding_value=0)
    
    # Create enhanced mel spectrograms
    mel_targets = []
    
    for item in batch:
        audio = item['audio']
        mel_spec = create_enhanced_mel(audio.unsqueeze(0))
        mel_targets.append(mel_spec.squeeze(0))
    
    # Pad mel spectrograms
    mel_padded = pad_sequence(mel_targets, batch_first=True, padding_value=-11.5)
    
    return {
        'text': text_padded,
        'text_lengths': torch.LongTensor(text_lengths),
        'mel_targets': mel_padded,
        'text_raw': [item['text_raw'] for item in batch]
    }


def train_full_yiddish():
    """Train on the complete Yiddish dataset"""
    print("=" * 70)
    print("🚀 FULL YIDDISH TTS TRAINING")
    print("=" * 70)
    print("• Training on complete dataset (272 samples)")
    print("• Enhanced architecture with attention")
    print("• Proven WaveGlow vocoder integration")
    print("• Target: Intelligible Yiddish speech")
    print("=" * 70)
    
    start_time = time.time()
    
    if not os.path.exists("tts_segments/segments_metadata.json"):
        print("❌ Error: Please ensure your data is in tts_segments/")
        return
    
    # Load complete dataset
    with open("tts_segments/segments_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    all_texts = [item['text'] for item in metadata]
    
    # Create tokenizer with full vocabulary
    tokenizer = YiddishTokenizer()
    tokenizer.build_vocab_from_texts(all_texts)
    
    # Save tokenizer
    tokenizer_data = {
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'vocab_size': tokenizer.vocab_size
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokenizer_file = f"yiddish_full_tokenizer_{timestamp}.json"
    
    with open(tokenizer_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer saved: {tokenizer_file}")
    
    # Create full dataset
    dataset = FullYiddishDataset(
        "tts_segments/segments_metadata.json",
        tokenizer,
        max_audio_length=10.0  # Allow longer audio clips
    )
    
    # Split dataset
    train_size = int(0.85 * len(dataset))  # Use more for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders - small batch size for memory efficiency
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_full)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_full)
    
    print(f"✅ Dataset split: {train_size} train, {val_size} validation")
    
    # Create enhanced model
    model = FullYiddishTacotron(vocab_size=tokenizer.vocab_size)
    
    device = torch.device('cpu')  # Keep on CPU for stability
    model.to(device)
    
    # Enhanced optimizer and scheduling
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )
    
    # Loss functions with proper weighting
    mel_criterion = nn.MSELoss()
    stop_criterion = nn.BCEWithLogitsLoss()
    
    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Extended training for speech quality
    num_epochs = 25  # More epochs for full learning
    best_loss = float('inf')
    
    print(f"🎯 Starting {num_epochs}-epoch training...")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        mel_loss_total = 0
        stop_loss_total = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            text = batch['text'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mel_outputs, mel_postnet, stop_outputs = model(text, mel_targets)
            
            # Align dimensions
            min_time = min(mel_outputs.size(1), mel_targets.size(1))
            mel_outputs_aligned = mel_outputs[:, :min_time, :]
            mel_postnet_aligned = mel_postnet[:, :min_time, :]
            mel_targets_aligned = mel_targets[:, :min_time, :]
            
            # Create stop targets
            stop_targets = torch.zeros(stop_outputs.size(0), min_time).to(device)
            stop_targets[:, -1] = 1.0  # Mark end
            
            # Calculate losses
            mel_loss = mel_criterion(mel_outputs_aligned, mel_targets_aligned)
            mel_postnet_loss = mel_criterion(mel_postnet_aligned, mel_targets_aligned)
            stop_loss = stop_criterion(stop_outputs[:, :min_time].squeeze(-1), stop_targets)
            
            # Combined loss with proper weighting
            loss = mel_loss + mel_postnet_loss + 0.5 * stop_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            mel_loss_total += (mel_loss + mel_postnet_loss).item()
            stop_loss_total += stop_loss.item()
            num_batches += 1
            
            # Progress updates
            if batch_idx % 20 == 0:
                elapsed = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}")
                print(f"  Loss: {loss.item():.4f} (Mel: {mel_loss.item():.4f}, PostNet: {mel_postnet_loss.item():.4f}, Stop: {stop_loss.item():.4f})")
                print(f"  Text: {batch['text_raw'][0][:50]}...")
                print(f"  Time: {elapsed:.1f}s")
        
        # Epoch summary
        avg_loss = total_loss / num_batches
        avg_mel_loss = mel_loss_total / num_batches
        avg_stop_loss = stop_loss_total / num_batches
        
        epoch_time = time.time() - epoch_start
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        print(f"\n✅ EPOCH {epoch+1} COMPLETED")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Mel Loss: {avg_mel_loss:.4f}")
        print(f"   Stop Loss: {avg_stop_loss:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Time: {epoch_time/60:.1f} minutes")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tokenizer': tokenizer_data,
                'loss': avg_loss,
                'timestamp': timestamp
            }
            best_model_file = f"yiddish_full_best_model_{timestamp}.pth"
            torch.save(best_checkpoint, best_model_file)
            print(f"   💾 New best model: {best_model_file} (loss: {avg_loss:.4f})")
        
        # Regular checkpoints
        if (epoch + 1) % 5 == 0:
            checkpoint_file = f"yiddish_full_checkpoint_epoch_{epoch+1}_{timestamp}.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'tokenizer': tokenizer_data,
                'loss': avg_loss,
                'timestamp': timestamp
            }
            torch.save(checkpoint, checkpoint_file)
            print(f"   💾 Checkpoint: {checkpoint_file}")
        
        print("-" * 50)
    
    total_time = time.time() - start_time
    
    print("=" * 70)
    print("🎉 FULL TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    print()
    print("🎯 Next steps:")
    print("1. Test speech generation with the best model")
    print("2. The model should now produce intelligible Yiddish speech")
    print("3. Use WaveGlow vocoder for audio generation")
    print("=" * 70)


if __name__ == "__main__":
    train_full_yiddish() 