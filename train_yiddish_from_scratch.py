#!/usr/bin/env python3
"""
Yiddish TTS Training From Scratch
Builds a TTS model from scratch with proper Hebrew character tokenization
"""

import os
import json
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
import unicodedata
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class YiddishTokenizer:
    """Custom tokenizer for Yiddish text (Hebrew script)"""
    
    def __init__(self):
        # Build character vocabulary from scratch
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
        print("Building vocabulary from Yiddish texts...")
        
        # Collect all unique characters
        all_chars = set()
        
        # Add special tokens first
        all_chars.update([self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN])
        
        # Add characters from texts
        for text in texts:
            # Normalize text
            normalized = self.normalize_text(text)
            all_chars.update(normalized)
        
        # Create mappings
        sorted_chars = sorted(list(all_chars))
        
        for idx, char in enumerate(sorted_chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        
        self.vocab_size = len(sorted_chars)
        
        print(f"Vocabulary built: {self.vocab_size} characters")
        hebrew_chars = [c for c in sorted_chars if len(c) == 1 and ord(c) >= 0x0590 and ord(c) <= 0x05FF]
        print(f"Hebrew characters found: {hebrew_chars}")
        
        return self.char_to_idx, self.idx_to_char
    
    def normalize_text(self, text):
        """Normalize Yiddish text"""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFD', text)
        
        # Remove combining marks but keep Hebrew characters
        normalized = ""
        for char in text:
            # Keep Hebrew script (0x0590-0x05FF)
            if 0x0590 <= ord(char) <= 0x05FF:
                normalized += char
            # Keep basic punctuation and spaces
            elif char in " .,!?;:-()[]{}\"'\n":
                normalized += char
            # Keep numbers
            elif char.isdigit():
                normalized += char
        
        # Clean up extra spaces
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return normalized
    
    def text_to_sequence(self, text):
        """Convert text to sequence of token indices"""
        normalized = self.normalize_text(text)
        sequence = [self.char_to_idx.get(self.START_TOKEN, 0)]
        
        for char in normalized:
            idx = self.char_to_idx.get(char, self.char_to_idx.get(self.UNK_TOKEN, 0))
            sequence.append(idx)
        
        sequence.append(self.char_to_idx.get(self.END_TOKEN, 0))
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        text = ""
        for idx in sequence:
            if idx < len(self.idx_to_char):
                char = self.idx_to_char[idx]
                if char not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                    text += char
        return text


class YiddishDataset(Dataset):
    """Dataset for Yiddish TTS training"""
    
    def __init__(self, metadata_file, audio_dir, text_dir, tokenizer, sample_rate=22050):
        self.audio_dir = Path(audio_dir)
        self.text_dir = Path(text_dir)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load audio (audio_file already contains full path)
        audio_path = Path(item['audio_file'])
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Tokenize text
        text_sequence = self.tokenizer.text_to_sequence(item['text'])
        
        return {
            'audio': audio.squeeze(),
            'text': torch.LongTensor(text_sequence),
            'text_raw': item['text'],
            'duration': item['duration']
        }


class SimpleTacotron(nn.Module):
    """Simplified Tacotron model for Yiddish TTS"""
    
    def __init__(self, vocab_size, embedding_dim=256, encoder_dim=256, decoder_dim=512, n_mels=80):
        super(SimpleTacotron, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.n_mels = n_mels
        
        # Text encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, encoder_dim, batch_first=True, bidirectional=True)
        
        # Attention mechanism - projects decoder input to match encoder dimensions
        self.input_projection = nn.Linear(n_mels, encoder_dim * 2)
        self.attention = nn.MultiheadAttention(encoder_dim * 2, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(encoder_dim * 2 + n_mels, decoder_dim, batch_first=True)
        self.mel_projection = nn.Linear(decoder_dim, n_mels)
        
        # Stop token predictor
        self.stop_projection = nn.Linear(decoder_dim, 1)
        
    def forward(self, text_sequences, mel_targets=None):
        batch_size = text_sequences.size(0)
        
        # Text encoding
        embedded = self.embedding(text_sequences)
        encoder_outputs, _ = self.encoder(embedded)
        
        if mel_targets is not None:
            # Training mode
            max_decoder_steps = mel_targets.size(1)
            decoder_outputs = []
            stop_tokens = []
            
            # Initialize decoder state
            decoder_input = torch.zeros(batch_size, 1, self.n_mels).to(text_sequences.device)
            hidden = None
            
            for step in range(max_decoder_steps):
                # Project decoder input to match encoder dimensions
                projected_input = self.input_projection(decoder_input)
                
                # Attention
                context, _ = self.attention(projected_input, encoder_outputs, encoder_outputs)
                
                # Concatenate context and previous mel
                decoder_input_concat = torch.cat([context, decoder_input], dim=-1)
                
                # Decoder step
                decoder_output, hidden = self.decoder_lstm(decoder_input_concat, hidden)
                
                # Project to mel and stop token
                mel_output = self.mel_projection(decoder_output)
                stop_output = self.stop_projection(decoder_output)
                
                decoder_outputs.append(mel_output)
                stop_tokens.append(stop_output)
                
                # Teacher forcing: use target mel for next input
                if step < max_decoder_steps - 1:
                    decoder_input = mel_targets[:, step:step+1, :]
            
            mel_outputs = torch.cat(decoder_outputs, dim=1)
            stop_outputs = torch.cat(stop_tokens, dim=1)
            
            return mel_outputs, stop_outputs
        else:
            # Inference mode
            max_decoder_steps = 1000  # Maximum generation length
            decoder_outputs = []
            stop_tokens = []
            
            decoder_input = torch.zeros(batch_size, 1, self.n_mels).to(text_sequences.device)
            hidden = None
            
            for step in range(max_decoder_steps):
                # Project decoder input to match encoder dimensions
                projected_input = self.input_projection(decoder_input)
                
                # Attention
                context, _ = self.attention(projected_input, encoder_outputs, encoder_outputs)
                
                # Concatenate context and previous mel
                decoder_input_concat = torch.cat([context, decoder_input], dim=-1)
                
                # Decoder step
                decoder_output, hidden = self.decoder_lstm(decoder_input_concat, hidden)
                
                # Project to mel and stop token
                mel_output = self.mel_projection(decoder_output)
                stop_output = self.stop_projection(decoder_output)
                
                decoder_outputs.append(mel_output)
                stop_tokens.append(stop_output)
                
                # Use generated mel for next input
                decoder_input = mel_output
                
                # Check stop condition
                if torch.sigmoid(stop_output).item() > 0.5:
                    break
            
            mel_outputs = torch.cat(decoder_outputs, dim=1)
            stop_outputs = torch.cat(stop_tokens, dim=1)
            
            return mel_outputs, stop_outputs


def collate_fn(batch):
    """Collate function for DataLoader"""
    # Sort by text length (longest first)
    batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
    
    # Pad sequences
    text_sequences = [item['text'] for item in batch]
    text_lengths = [len(seq) for seq in text_sequences]
    text_padded = pad_sequence(text_sequences, batch_first=True, padding_value=0)
    
    # For now, we'll create dummy mel spectrograms
    # In a full implementation, you'd convert audio to mel spectrograms
    max_audio_len = max([item['audio'].size(0) for item in batch])
    n_mels = 80
    
    mel_targets = []
    for item in batch:
        # Convert audio to mel spectrogram
        audio = item['audio']
        
        # Create mel spectrogram from audio
        mel_spec = create_mel_spectrograms(audio.unsqueeze(0))
        mel_targets.append(mel_spec.squeeze(0))
    
    # Pad mel spectrograms
    mel_padded = pad_sequence(mel_targets, batch_first=True, padding_value=0)
    
    return {
        'text': text_padded,
        'text_lengths': torch.LongTensor(text_lengths),
        'mel_targets': mel_padded,
        'text_raw': [item['text_raw'] for item in batch]
    }


def create_mel_spectrograms(audio_tensor, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256):
    """Convert audio to mel spectrograms"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    
    mel_spec = mel_transform(audio_tensor)
    mel_spec = torch.log(mel_spec + 1e-8)  # Log mel spectrogram
    return mel_spec.transpose(-1, -2)  # (time, n_mels)


def train_yiddish_tts():
    """Main training function"""
    print("=== Training Yiddish TTS From Scratch ===")
    print("Building custom tokenizer with Hebrew characters...")
    
    # Check data
    if not os.path.exists("tts_segments/segments_metadata.json"):
        print("Error: Please ensure your data is in tts_segments/")
        return
    
    # Load all texts to build vocabulary
    with open("tts_segments/segments_metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    all_texts = [item['text'] for item in metadata]
    
    # Create tokenizer and build vocabulary
    tokenizer = YiddishTokenizer()
    tokenizer.build_vocab_from_texts(all_texts)
    
    # Save tokenizer
    tokenizer_data = {
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'vocab_size': tokenizer.vocab_size
    }
    
    with open("yiddish_tokenizer.json", 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Tokenizer saved with {tokenizer.vocab_size} characters")
    
    # Create dataset
    dataset = YiddishDataset(
        "tts_segments/segments_metadata.json",
        "tts_segments/audio",
        "tts_segments/text",
        tokenizer
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print(f"✅ Dataset split: {train_size} train, {val_size} validation")
    
    # Create model
    model = SimpleTacotron(vocab_size=tokenizer.vocab_size)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"✅ Model created, training on {device}")
    
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            text = batch['text'].to(device)
            mel_targets = batch['mel_targets'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            mel_outputs, stop_outputs = model(text, mel_targets)
            
            # Calculate loss
            mel_loss = criterion(mel_outputs, mel_targets)
            loss = mel_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer_data,
                'loss': avg_loss
            }
            torch.save(checkpoint, f"yiddish_tts_checkpoint_epoch_{epoch+1}.pth")
            print(f"✅ Checkpoint saved: epoch {epoch+1}")
    
    print("🎉 Training completed!")
    print("Next steps:")
    print("1. The model now understands Hebrew characters properly")
    print("2. You can generate speech using the trained model")
    print("3. Fine-tune with more epochs if needed")


if __name__ == "__main__":
    train_yiddish_tts() 