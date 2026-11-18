import os
import json
import time
import math
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Optimize CPU threads for PyTorch
torch.set_num_threads(4)

from dataset.preprocess import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from models.seq2seq_attention import EncoderLSTM, DecoderWithAttention, Seq2SeqAttention
from models.seq2seq_baseline import EncoderBaseline, DecoderBaseline, Seq2SeqBaseline
from training.bleu import calculate_sentence_bleu, calculate_corpus_bleu

class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Dict], src_vocab: Vocabulary, trg_vocab: Vocabulary):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_indices = self.src_vocab.sentence_to_indices(pair["src_tokens"], add_sos_eos=True)
        trg_indices = self.trg_vocab.sentence_to_indices(pair["trg_tokens"], add_sos_eos=True)
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long), pair


def collate_fn(batch):
    src_tensors, trg_tensors, pairs = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=PAD_IDX)
    trg_padded = nn.utils.rnn.pad_sequence(trg_tensors, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded, pairs


def train_epoch_attention(model: Seq2SeqAttention, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                          criterion: nn.CrossEntropyLoss, clip: float = 1.0, device: torch.device = torch.device("cpu")):
    model.train()
    epoch_loss = 0.0

    for src, trg, _ in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_target = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_attention(model: Seq2SeqAttention, dataloader: DataLoader, criterion: nn.CrossEntropyLoss,
                       trg_vocab: Vocabulary, device: torch.device = torch.device("cpu"), max_eval_samples: int = 40):
    model.eval()
    epoch_loss = 0.0
    references = []
    candidates = []

    with torch.no_grad():
        for src, trg, pairs in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            loss_output = output[:, 1:].reshape(-1, output_dim)
            loss_trg = trg[:, 1:].reshape(-1)

            loss = criterion(loss_output, loss_trg)
            epoch_loss += loss.item()

            if len(candidates) < max_eval_samples:
                for b_idx in range(min(src.size(0), max_eval_samples - len(candidates))):
                    pair = pairs[b_idx]
                    src_clean = [idx for idx in src[b_idx].tolist() if idx not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                    pred_indices, _ = model.translate(src_clean, device=device, max_len=15)
                    pred_tokens = trg_vocab.indices_to_tokens(pred_indices, remove_special=True)
                    references.append(pair["trg_tokens"])
                    candidates.append(pred_tokens)

    val_loss = epoch_loss / len(dataloader)
    val_bleu = calculate_corpus_bleu(references, candidates)
    return val_loss, val_bleu


def train_epoch_baseline(model: Seq2SeqBaseline, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                         criterion: nn.CrossEntropyLoss, clip: float = 1.0, device: torch.device = torch.device("cpu")):
    model.train()
    epoch_loss = 0.0

    for src, trg, _ in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio=0.5)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_target = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg_target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_baseline(model: Seq2SeqBaseline, dataloader: DataLoader, criterion: nn.CrossEntropyLoss,
                      trg_vocab: Vocabulary, device: torch.device = torch.device("cpu"), max_eval_samples: int = 40):
    model.eval()
    epoch_loss = 0.0
    references = []
    candidates = []

    with torch.no_grad():
        for src, trg, pairs in dataloader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            loss_output = output[:, 1:].reshape(-1, output_dim)
            loss_trg = trg[:, 1:].reshape(-1)

            loss = criterion(loss_output, loss_trg)
            epoch_loss += loss.item()

            if len(candidates) < max_eval_samples:
                for b_idx in range(min(src.size(0), max_eval_samples - len(candidates))):
                    pair = pairs[b_idx]
                    src_clean = [idx for idx in src[b_idx].tolist() if idx not in (PAD_IDX, SOS_IDX, EOS_IDX)]
                    pred_indices = model.translate(src_clean, device=device, max_len=15)
                    pred_tokens = trg_vocab.indices_to_tokens(pred_indices, remove_special=True)
                    references.append(pair["trg_tokens"])
                    candidates.append(pred_tokens)

    val_loss = epoch_loss / len(dataloader)
    val_bleu = calculate_corpus_bleu(references, candidates)
    return val_loss, val_bleu


def run_training(epochs: int = 4, batch_size: int = 128, emb_dim: int = 128, hidden_dim: int = 256, attn_dim: int = 128):
    os.makedirs("models/checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for training: {device}", flush=True)

    # Load vocabs
    with open("dataset/src_vocab.json", "r", encoding="utf-8") as f:
        src_vocab = Vocabulary.from_dict(json.load(f))
    with open("dataset/trg_vocab.json", "r", encoding="utf-8") as f:
        trg_vocab = Vocabulary.from_dict(json.load(f))

    # Load pairs (10,800 train pairs > 10,000 requirement)
    with open("dataset/train_pairs.json", "r", encoding="utf-8") as f:
        train_pairs = json.load(f)
    with open("dataset/val_pairs.json", "r", encoding="utf-8") as f:
        val_pairs = json.load(f)

    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.", flush=True)
    print(f"Vocab sizes - Source: {src_vocab.n_words}, Target: {trg_vocab.n_words}", flush=True)

    train_dataset = TranslationDataset(train_pairs, src_vocab, trg_vocab)
    val_dataset = TranslationDataset(val_pairs, src_vocab, trg_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 1. Initialize Attention Model (Model B)
    print("Initializing Model B: Seq2Seq with Bahdanau Attention...", flush=True)
    enc_attn = EncoderLSTM(src_vocab.n_words, emb_dim, hidden_dim, dropout=0.2, padding_idx=PAD_IDX)
    dec_attn = DecoderWithAttention(trg_vocab.n_words, emb_dim, hidden_dim * 2, hidden_dim, attn_dim, dropout=0.2, padding_idx=PAD_IDX)
    model_attention = Seq2SeqAttention(enc_attn, dec_attn, pad_idx=PAD_IDX).to(device)
    optimizer_attn = torch.optim.Adam(model_attention.parameters(), lr=0.003)

    attn_param_count = sum(p.numel() for p in model_attention.parameters() if p.requires_grad)
    print(f"Model B Parameter Count: {attn_param_count:,}", flush=True)

    # 2. Initialize Baseline Model (Model A)
    print("Initializing Model A: Baseline Seq2Seq (Without Attention)...", flush=True)
    enc_base = EncoderBaseline(src_vocab.n_words, emb_dim, hidden_dim, dropout=0.2, padding_idx=PAD_IDX)
    dec_base = DecoderBaseline(trg_vocab.n_words, emb_dim, hidden_dim, dropout=0.2, padding_idx=PAD_IDX)
    model_baseline = Seq2SeqBaseline(enc_base, dec_base, pad_idx=PAD_IDX).to(device)
    optimizer_base = torch.optim.Adam(model_baseline.parameters(), lr=0.003)

    base_param_count = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    print(f"Model A Parameter Count: {base_param_count:,}", flush=True)

    history = {
        "epochs": [],
        "attention_model": {
            "name": "Seq2Seq with Bahdanau Attention",
            "train_loss": [],
            "val_loss": [],
            "val_bleu": [],
            "parameters": attn_param_count
        },
        "baseline_model": {
            "name": "Seq2Seq Baseline (No Attention)",
            "train_loss": [],
            "val_loss": [],
            "val_bleu": [],
            "parameters": base_param_count
        },
        "training_time_seconds": 0,
        "dataset_info": {
            "train_size": len(train_pairs),
            "val_size": len(val_pairs),
            "src_vocab_size": src_vocab.n_words,
            "trg_vocab_size": trg_vocab.n_words
        }
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        print(f"\n--- Starting Epoch {epoch}/{epochs} ---", flush=True)

        # Train Attention Model
        train_loss_attn = train_epoch_attention(model_attention, train_loader, optimizer_attn, criterion, device=device)
        val_loss_attn, val_bleu_attn = evaluate_attention(model_attention, val_loader, criterion, trg_vocab, device=device)

        # Train Baseline Model
        train_loss_base = train_epoch_baseline(model_baseline, train_loader, optimizer_base, criterion, device=device)
        val_loss_base, val_bleu_base = evaluate_baseline(model_baseline, val_loader, criterion, trg_vocab, device=device)

        ep_duration = round(time.time() - ep_start, 1)

        history["epochs"].append(epoch)
        history["attention_model"]["train_loss"].append(round(train_loss_attn, 4))
        history["attention_model"]["val_loss"].append(round(val_loss_attn, 4))
        history["attention_model"]["val_bleu"].append(val_bleu_attn)

        history["baseline_model"]["train_loss"].append(round(train_loss_base, 4))
        history["baseline_model"]["val_loss"].append(round(val_loss_base, 4))
        history["baseline_model"]["val_bleu"].append(val_bleu_base)

        print(f"Epoch {epoch} finished in {ep_duration}s", flush=True)
        print(f" [Model B - Attention] Train Loss: {train_loss_attn:.4f} | Val Loss: {val_loss_attn:.4f} | Val BLEU: {val_bleu_attn:.2f}", flush=True)
        print(f" [Model A - Baseline ] Train Loss: {train_loss_base:.4f} | Val Loss: {val_loss_base:.4f} | Val BLEU: {val_bleu_base:.2f}", flush=True)

        # Save incremental checkpoint after each epoch!
        torch.save(model_attention.state_dict(), "models/checkpoints/seq2seq_attention.pt")
        torch.save(model_baseline.state_dict(), "models/checkpoints/seq2seq_baseline.pt")
        with open("models/checkpoints/training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = round(time.time() - start_time, 1)
    history["training_time_seconds"] = total_time
    print(f"\nAll {epochs} Epochs Completed in {total_time}s!", flush=True)

    config = {
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "attn_dim": attn_dim,
        "src_vocab_size": src_vocab.n_words,
        "trg_vocab_size": trg_vocab.n_words,
        "pad_idx": PAD_IDX,
        "sos_idx": SOS_IDX,
        "eos_idx": EOS_IDX,
        "unk_idx": UNK_IDX,
        "architecture_attention": "Bidirectional LSTM Encoder (2x256) + Bahdanau Additive Attention + LSTM Decoder (256)",
        "architecture_baseline": "Unidirectional LSTM Encoder (256) + LSTM Decoder (256) without Attention",
        "last_trained": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("models/checkpoints/model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open("models/checkpoints/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Checkpoints, config, and training history saved to models/checkpoints/", flush=True)
    return history


if __name__ == "__main__":
    run_training(epochs=4, batch_size=128)
