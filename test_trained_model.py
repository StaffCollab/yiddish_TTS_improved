#!/usr/bin/env python3
"""
Test Your Trained Yiddish Model!
Load the conservative checkpoint and generate Yiddish speech
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

# EXACT architecture from train_full_yiddish_conservative.py
class YiddishTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = ' 