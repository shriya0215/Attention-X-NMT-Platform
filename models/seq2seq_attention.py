import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.2, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Project bidirectional hidden state to decoder hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        encoder_outputs, (hidden, cell) = self.lstm(embedded)
        # encoder_outputs: [batch_size, src_len, hidden_dim * 2]
        # hidden, cell: [2, batch_size, hidden_dim]

        # Concatenate forward and backward final hidden states
        h_cat = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch_size, hidden_dim * 2]
        c_cat = torch.cat((cell[0], cell[1]), dim=1)

        dec_init_hidden = torch.tanh(self.fc_hidden(h_cat))  # [batch_size, hidden_dim]
        dec_init_cell = torch.tanh(self.fc_cell(c_cat))      # [batch_size, hidden_dim]

        return encoder_outputs, dec_init_hidden, dec_init_cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism:
    score(s_{t-1}, h_i) = v^T * tanh(W_s * s_{t-1} + W_h * h_i)
    alpha_{t, i} = softmax(score(s_{t-1}, h_i))
    context_t = sum(alpha_{t, i} * h_i)
    """
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, attn_dim: int):
        super().__init__()
        # enc_hidden_dim is hidden_dim * 2 for bidirectional encoder
        self.W_h = nn.Linear(enc_hidden_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, dec_hidden: torch.Tensor, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # dec_hidden: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim]

        src_len = encoder_outputs.size(1)

        # Expand dec_hidden to match source length: [batch_size, src_len, dec_hidden_dim]
        dec_hidden_expanded = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Calculate alignment energy scores
        wh = self.W_h(encoder_outputs)          # [batch_size, src_len, attn_dim]
        ws = self.W_s(dec_hidden_expanded)      # [batch_size, src_len, attn_dim]
        energy = torch.tanh(wh + ws)            # [batch_size, src_len, attn_dim]

        # Score vector
        scores = self.v(energy).squeeze(2)      # [batch_size, src_len]

        # Apply mask if padding exists
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax gives the attention distribution alpha
        attn_weights = F.softmax(scores, dim=1)  # [batch_size, src_len]

        # Context vector: weighted sum of encoder outputs
        # [batch_size, 1, src_len] bmm [batch_size, src_len, enc_hidden_dim] -> [batch_size, 1, enc_hidden_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [batch_size, enc_hidden_dim]

        return context, attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, enc_hidden_dim: int, dec_hidden_dim: int,
                 attn_dim: int, dropout: float = 0.2, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(enc_hidden_dim, dec_hidden_dim, attn_dim)
        
        # LSTM input is combination of target word embedding and attention context vector
        self.lstm_cell = nn.LSTMCell(emb_dim + enc_hidden_dim, dec_hidden_dim)
        
        # Output layer projects concatenation of: [current embedding; decoder hidden; attention context]
        self.fc_out = nn.Linear(emb_dim + dec_hidden_dim + enc_hidden_dim, vocab_size)

    def forward_step(self, token: torch.Tensor, dec_hidden: torch.Tensor, dec_cell: torch.Tensor,
                     encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # token: [batch_size]
        # dec_hidden, dec_cell: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, src_len, enc_hidden_dim]

        embedded = self.dropout(self.embedding(token))  # [batch_size, emb_dim]

        # 1. Compute Bahdanau Attention
        context, attn_weights = self.attention(dec_hidden, encoder_outputs, mask)

        # 2. Step LSTM
        lstm_input = torch.cat((embedded, context), dim=1)  # [batch_size, emb_dim + enc_hidden_dim]
        new_hidden, new_cell = self.lstm_cell(lstm_input, (dec_hidden, dec_cell))

        # 3. Output prediction
        out_input = torch.cat((embedded, new_hidden, context), dim=1)
        logits = self.fc_out(out_input)  # [batch_size, vocab_size]

        return logits, new_hidden, new_cell, attn_weights


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderWithAttention, pad_idx: int = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        # mask is 1 where src != pad_idx, 0 where padding
        return (src != self.pad_idx).float()

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=src.device)
        mask = self.create_mask(src)

        encoder_outputs, dec_hidden, dec_cell = self.encoder(src)

        # First input token is <sos> at index 0 of trg
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            logits, dec_hidden, dec_cell, _ = self.decoder.forward_step(
                input_token, dec_hidden, dec_cell, encoder_outputs, mask
            )
            outputs[:, t, :] = logits
            
            # Teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_token = trg[:, t] if teacher_force else top1

        return outputs

    def translate(self, src_indices: List[int], sos_idx: int = 1, eos_idx: int = 2, max_len: int = 30, device: torch.device = torch.device("cpu")):
        self.eval()
        with torch.no_grad():
            src_tensor = torch.tensor([src_indices], dtype=torch.long, device=device)
            mask = self.create_mask(src_tensor)

            encoder_outputs, dec_hidden, dec_cell = self.encoder(src_tensor)

            current_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
            predicted_indices = []
            attention_matrix = []

            for _ in range(max_len):
                logits, dec_hidden, dec_cell, attn_weights = self.decoder.forward_step(
                    current_token, dec_hidden, dec_cell, encoder_outputs, mask
                )
                
                # attn_weights: [1, src_len]
                attention_matrix.append(attn_weights.squeeze(0).cpu().tolist())

                top1 = logits.argmax(1).item()
                if top1 == eos_idx:
                    break
                predicted_indices.append(top1)
                current_token = torch.tensor([top1], dtype=torch.long, device=device)

            return predicted_indices, attention_matrix
