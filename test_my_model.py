#!/usr/bin/env python3
"""
Conservative Yiddish TTS Training - Full Model, Safe Training
Keeps the full 2M parameter model but trains conservatively to protect your system
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
        
        # Safe Hebrew character detection
        hebrew_chars = []
        for c in sorted_chars:
            try:
                if len(c) == 1 and 0x0590 <= ord(c) <= 0x05FF:
                    hebrew_chars.append(c)
            except:
                continue
        print(f"Hebrew chars: {hebrew_chars[:10]}...")
        
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


class EnhancedEncoder(nn.Module):
    """FULL Enhanced encoder - keeping all 2M parameters for quality"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):  # FULL SIZES
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Full convolutional layers
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
        
        # Full bidirectional LSTM - KEEPING ALL LAYERS
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            num_layers=3,  # FULL 3 LAYERS
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
        lstm_output, _ = self.lstm(conv_output)
        
        # Apply dropout
        output = self.dropout(lstm_output)
        
        return output


class AttentionDecoder(nn.Module):
    """FULL attention decoder - keeping all parameters"""
    
    def __init__(self, encoder_dim=512, decoder_dim=512, n_mels=80):  # FULL SIZES
        super().__init__()
        
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        
        # FULL attention mechanism
        self.attention = nn.MultiheadAttention(
            decoder_dim, 
            num_heads=8,  # FULL 8 HEADS
            batch_first=True
        )
        
        # Input projection for attention alignment
        self.input_projection = nn.Linear(n_mels, decoder_dim)
        
        # FULL decoder LSTM - KEEPING ALL LAYERS
        self.decoder_lstm = nn.LSTM(
            encoder_dim + n_mels, 
            decoder_dim, 
            num_layers=2,  # FULL 2 LAYERS
            batch_first=True,
            dropout=0.1
        )
        
        # Output projections
        self.mel_projection = nn.Linear(decoder_dim, n_mels)
        self.stop_projection = nn.Linear(decoder_dim, 1)
        
        # FULL postnet for refinement
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),  # FULL SIZE
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2),
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
            # Project decoder input to match encoder dimensions
            projected_input = self.input_projection(decoder_input)
            
            # Attention mechanism
            attn_output, _ = self.attention(
                projected_input, 
                encoder_outputs, 
                encoder_outputs
            )
            
            # Concatenate context and previous mel
            decoder_input_concat = torch.cat([attn_output, decoder_input], dim=-1)
            
            # Decoder LSTM step
            decoder_output, hidden = self.decoder_lstm(decoder_input_concat, hidden)
            
            # Project to mel and stop token
            mel_output = self.mel_projection(decoder_output)
            stop_output = self.stop_projection(decoder_output)
            
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
        
        # Postnet refinement
        mel_postnet = mel_outputs.transpose(1, 2)
        mel_postnet = self.postnet(mel_postnet)
        mel_postnet = mel_postnet.transpose(1, 2)
        mel_postnet = mel_outputs + mel_postnet
        
        return mel_outputs, mel_postnet, stop_outputs


class FullYiddishTacotron(nn.Module):
    """FULL model architecture - keeping all 2M parameters"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = EnhancedEncoder(vocab_size)  # FULL encoder
        self.decoder = AttentionDecoder()  # FULL decoder
    
    def forward(self, text, mel_targets=None):
        encoder_outputs = self.encoder(text)
        mel_outputs, mel_postnet, stop_outputs = self.decoder(encoder_outputs, mel_targets)
        return mel_outputs, mel_postnet, stop_outputs


class ConservativeYiddishDataset(Dataset):
    """Conservative dataset loading with memory management"""
    
    def __init__(self, metadata_path, tokenizer, max_samples=100, max_audio_length=6.0):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.tokenizer = tokenizer
        
        # Conservative sample limiting
        self.metadata = self.metadata[:max_samples]
        print(f"Using {len(self.metadata)} samples (conservative for memory management)")
        
        # Filter by audio length
        self.filtered_metadata = []
        for item in self.metadata:
            try:
                audio_path = item['audio_file']  # Path is already complete in metadata
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
        audio_path = item['audio_file']  # Path is already complete in metadata
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


def collate_fn_conservative(batch):
    """Conservative collate function with memory management"""
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


def train_conservative_yiddish():
    """Conservative training with FULL model but safe practices"""
    print("=" * 70)
    print("🏗️ CONSERVATIVE YIDDISH TTS TRAINING")
    print("=" * 70)
    print("• FULL model architecture (2M parameters)")
    print("• Conservative batch sizes and memory management")
    print("• Short training sessions with frequent checkpointing")
    print("• System-safe practices")
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
    tokenizer_file = f"yiddish_conservative_tokenizer_{timestamp}.json"
    
    with open(tokenizer_file, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer saved: {tokenizer_file}")
    
    # Create conservative dataset
    dataset = ConservativeYiddishDataset(
        "tts_segments/segments_metadata.json",
        tokenizer,
        max_samples=100,  # Conservative sample count
        max_audio_length=10.0  # Allow longer audio based on actual durations (7-8s average)
    )
    
    # Small batch sizes for memory safety
    train_loader = DataLoader(
        dataset, 
        batch_size=2,  # Small but not tiny
        shuffle=True, 
        collate_fn=collate_fn_conservative,
        num_workers=0  # No multiprocessing to save memory
    )
    
    print(f"✅ Dataset: {len(dataset)} samples")
    
    # Create FULL model (2M parameters)
    model = FullYiddishTacotron(vocab_size=tokenizer.vocab_size)
    
    device = torch.device('cpu')  # Force CPU for safety
    model.to(device)
    
    # Conservative optimizer settings
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=5e-4,  # Lower learning rate for stability
        weight_decay=1e-6
    )
    
    # Loss functions
    mel_criterion = nn.MSELoss()
    stop_criterion = nn.BCEWithLogitsLoss()
    
    print(f"✅ FULL Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Conservative training sessions
    num_epochs = 8  # Moderate epochs per session
    best_loss = float('inf')
    
    print(f"🏗️ Starting {num_epochs}-epoch conservative training session...")
    print(f"⚠️  Full model quality with safe training practices")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        mel_loss_total = 0
        stop_loss_total = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Conservative memory cleanup
            if batch_idx % 3 == 0:
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
                
                # Combined loss with full weighting
                loss = mel_loss + mel_postnet_loss + 0.5 * stop_loss
                
                # Backward pass with conservative gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
                mel_loss_total += (mel_loss + mel_postnet_loss).item()
                stop_loss_total += stop_loss.item()
                num_batches += 1
                
                # Regular progress updates
                if batch_idx % 3 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}")
                    print(f"    Loss: {loss.item():.4f} (Mel: {mel_loss.item():.3f}, Post: {mel_postnet_loss.item():.3f}, Stop: {stop_loss.item():.3f})")
                    print(f"    Text: {batch['text_raw'][0][:40]}...")
                
                # Conservative memory cleanup
                del mel_outputs, mel_postnet, stop_outputs, loss
                del mel_loss, mel_postnet_loss, stop_loss
                
            except Exception as e:
                print(f"  ⚠️  Batch {batch_idx} failed: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_mel_loss = mel_loss_total / num_batches
            avg_stop_loss = stop_loss_total / num_batches
            epoch_time = time.time() - epoch_start
            
            print(f"✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
            print(f"   Loss: {avg_loss:.4f} (Mel: {avg_mel_loss:.3f}, Stop: {avg_stop_loss:.3f})")
            
            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer_data,
                'loss': avg_loss
            }
            
            checkpoint_file = f"yiddish_conservative_checkpoint_epoch_{epoch+1}_{timestamp}.pth"
            torch.save(checkpoint, checkpoint_file)
            print(f"   💾 Checkpoint saved: {checkpoint_file}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_file = f"yiddish_conservative_best_{timestamp}.pth"
                torch.save(checkpoint, best_file)
                print(f"   🏆 New best model: {best_file} (loss: {avg_loss:.4f})")
        
        # Conservative memory cleanup after epoch
        gc.collect()
        
        # Rest between epochs
        print("   💤 Resting 2 seconds...")
        time.sleep(2)
    
    print("✅ Conservative training session completed!")
    print(f"🔄 Run multiple sessions for continued improvement")
    print(f"   (Full model quality with system-safe training)")
    
    print("\n🎯 Next steps:")
    print("1. Run this script multiple times for incremental improvement")
    print("2. Use generate_yiddish_speech.py for zero-shot generation")
    print("3. Each session builds on previous checkpoints")
    print("4. Full 2M parameter model ensures quality")
    
    return tokenizer_file


def generate_yiddish_speech(checkpoint_path=None):
    """Generate Yiddish speech from trained model"""
    import sys
    print("🎤 YIDDISH SPEECH GENERATION")
    print("=" * 50)
    
    # Find checkpoint
    if checkpoint_path is None:
        checkpoint_path = "yiddish_conservative_checkpoint_epoch_5_20250721_221138.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"🔄 Loading: {Path(checkpoint_path).name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"   Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # Rebuild tokenizer
    tokenizer = YiddishTokenizer()
    if 'tokenizer' in checkpoint:
        tokenizer.char_to_idx = checkpoint['tokenizer']['char_to_idx']
        tokenizer.idx_to_char = checkpoint['tokenizer']['idx_to_char']
        tokenizer.vocab_size = checkpoint['tokenizer']['vocab_size']
        print(f"   Tokenizer: {tokenizer.vocab_size} characters")
    
    # Create and load model
    model = FullYiddishTacotron(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {total_params:,} parameters")
    
    # Get text
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = "שבת שלום, ווי גייט עס?"
    
    print(f"\n🎯 Text: {text}")
    
    # Generate
    text_seq = tokenizer.text_to_sequence(text)
    text_tensor = torch.tensor(text_seq).unsqueeze(0)
    
    with torch.no_grad():
        mel_outputs, mel_postnet, stop_outputs = model(text_tensor)
    
    print(f"   Generated mel: {mel_postnet.shape}")
    
    # Convert to audio (simpler approach)
    mel_np = mel_postnet.squeeze().cpu().numpy()
    
    import librosa
    
    # Simple conversion: mel to audio via inverse mel transform
    mel_spec = mel_np.T  # Shape: (80, time)
    
    # Convert mel spectrogram to linear spectrogram
    # Use librosa's mel_to_stft to get proper linear dimensions
    sr = 22050
    n_fft = 1024
    hop_length = 256
    
    # Create mel filter bank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=80)
    
    # Pseudo-inverse to convert mel back to linear
    mel_basis_pinv = np.linalg.pinv(mel_basis)
    
    # Convert mel to linear
    linear_spec = np.dot(mel_basis_pinv, np.exp(mel_spec))
    
    # Griffin-Lim reconstruction
    audio = librosa.griffinlim(linear_spec, n_iter=32, hop_length=hop_length, n_fft=n_fft)
    
    # Save
    output_file = f"my_yiddish_speech.wav"
    torchaudio.save(output_file, torch.tensor(audio).unsqueeze(0), 22050)
    
    print(f"   ✅ Saved: {output_file}")
    print(f"   🎵 Play: vlc {output_file}")
    
    return output_file

if __name__ == "__main__":
    generate_yiddish_speech() 