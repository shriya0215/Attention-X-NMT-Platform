import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class EncoderBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.2, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        _, (hidden, cell) = self.lstm(embedded)
        # hidden, cell: [1, batch_size, hidden_dim]
        return hidden, cell


class DecoderBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.2, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_token: [batch_size, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch_size, 1, hidden_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, vocab_size]
        return prediction, hidden, cell


class Seq2SeqBaseline(nn.Module):
    def __init__(self, encoder: EncoderBaseline, decoder: DecoderBaseline, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        hidden, cell = self.encoder(src)

        input_token = trg[:, 0].unsqueeze(1)  # [batch_size, 1]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t, :] = prediction

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1)
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def translate(self, src_indices: List[int], sos_idx: int = 1, eos_idx: int = 2, max_len: int = 30, device: torch.device = torch.device("cpu")):
        self.eval()
        with torch.no_grad():
            src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
            hidden, cell = self.encoder(src_tensor)

            current_token = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
            predicted_indices = []

            for _ in range(max_len):
                prediction, hidden, cell = self.decoder(current_token, hidden, cell)
                top1 = prediction.argmax(1).item()
                if top1 == eos_idx:
                    break
                predicted_indices.append(top1)
                current_token = torch.tensor([[top1]], dtype=torch.long, device=device)

            return predicted_indices
