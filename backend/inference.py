import os
import json
import time
import math
import torch
from typing import Dict, List, Any, Optional

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.preprocess import Vocabulary, tokenize_english, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from models.seq2seq_attention import EncoderLSTM, DecoderWithAttention, Seq2SeqAttention
from models.seq2seq_baseline import EncoderBaseline, DecoderBaseline, Seq2SeqBaseline

LOGS_FILE = os.path.join(os.path.dirname(__file__), "translation_logs.json")

class InferenceEngine:
    def __init__(self, models_dir: str = "models/checkpoints", dataset_dir: str = "dataset"):
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.src_vocab: Optional[Vocabulary] = None
        self.trg_vocab: Optional[Vocabulary] = None
        self.model_attention: Optional[Seq2SeqAttention] = None
        self.model_baseline: Optional[Seq2SeqBaseline] = None
        self.config: Dict[str, Any] = {}
        self.dataset_meta: Dict[str, Any] = {}
        self.training_history: Dict[str, Any] = {}
        
        self.load_resources()
        self._init_logs()

    def _init_logs(self):
        if not os.path.exists(LOGS_FILE):
            initial_logs = {
                "total_translations": 0,
                "total_latency_ms": 0.0,
                "history": []
            }
            with open(LOGS_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_logs, f, indent=2)

    def log_translation(self, src_text: str, trg_text: str, latency_ms: float, model_type: str):
        try:
            with open(LOGS_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
            logs["total_translations"] += 1
            logs["total_latency_ms"] += latency_ms
            logs["history"].append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "src_text": src_text,
                "trg_text": trg_text,
                "latency_ms": round(latency_ms, 2),
                "model_type": model_type
            })
            # Keep last 100 history entries
            if len(logs["history"]) > 100:
                logs["history"] = logs["history"][-100:]
            with open(LOGS_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error logging translation: {e}")

    def load_resources(self):
        # 1. Load Vocabularies
        src_vocab_path = os.path.join(self.dataset_dir, "src_vocab.json")
        trg_vocab_path = os.path.join(self.dataset_dir, "trg_vocab.json")
        if os.path.exists(src_vocab_path) and os.path.exists(trg_vocab_path):
            with open(src_vocab_path, "r", encoding="utf-8") as f:
                self.src_vocab = Vocabulary.from_dict(json.load(f))
            with open(trg_vocab_path, "r", encoding="utf-8") as f:
                self.trg_vocab = Vocabulary.from_dict(json.load(f))

        # 2. Load Metadata & Config
        config_path = os.path.join(self.models_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            # Fallback default configuration
            self.config = {
                "emb_dim": 128,
                "hidden_dim": 256,
                "attn_dim": 128,
                "src_vocab_size": self.src_vocab.n_words if self.src_vocab else 3544,
                "trg_vocab_size": self.trg_vocab.n_words if self.trg_vocab else 6000,
                "pad_idx": PAD_IDX,
                "sos_idx": SOS_IDX,
                "eos_idx": EOS_IDX,
                "unk_idx": UNK_IDX,
                "architecture_attention": "Bidirectional LSTM Encoder (2x256) + Bahdanau Additive Attention + LSTM Decoder (256)",
                "architecture_baseline": "Unidirectional LSTM Encoder (256) + LSTM Decoder (256) without Attention"
            }

        dataset_meta_path = os.path.join(self.dataset_dir, "dataset_metadata.json")
        if os.path.exists(dataset_meta_path):
            with open(dataset_meta_path, "r", encoding="utf-8") as f:
                self.dataset_meta = json.load(f)

        history_path = os.path.join(self.models_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                self.training_history = json.load(f)

        # 3. Instantiate Models
        if self.src_vocab and self.trg_vocab:
            emb_dim = self.config.get("emb_dim", 128)
            hidden_dim = self.config.get("hidden_dim", 256)
            attn_dim = self.config.get("attn_dim", 128)

            # Attention Model
            enc_attn = EncoderLSTM(self.src_vocab.n_words, emb_dim, hidden_dim, dropout=0.0, padding_idx=PAD_IDX)
            dec_attn = DecoderWithAttention(self.trg_vocab.n_words, emb_dim, hidden_dim * 2, hidden_dim, attn_dim, dropout=0.0, padding_idx=PAD_IDX)
            self.model_attention = Seq2SeqAttention(enc_attn, dec_attn, pad_idx=PAD_IDX).to(self.device)

            # Baseline Model
            enc_base = EncoderBaseline(self.src_vocab.n_words, emb_dim, hidden_dim, dropout=0.0, padding_idx=PAD_IDX)
            dec_base = DecoderBaseline(self.trg_vocab.n_words, emb_dim, hidden_dim, dropout=0.0, padding_idx=PAD_IDX)
            self.model_baseline = Seq2SeqBaseline(enc_base, dec_base, pad_idx=PAD_IDX).to(self.device)

            # Load Checkpoints if they exist
            attn_ckpt = os.path.join(self.models_dir, "seq2seq_attention.pt")
            if os.path.exists(attn_ckpt):
                self.model_attention.load_state_dict(torch.load(attn_ckpt, map_location=self.device))
                self.model_attention.eval()
                print("Loaded trained Seq2Seq Attention weights successfully.")

            base_ckpt = os.path.join(self.models_dir, "seq2seq_baseline.pt")
            if os.path.exists(base_ckpt):
                self.model_baseline.load_state_dict(torch.load(base_ckpt, map_location=self.device))
                self.model_baseline.eval()
                print("Loaded trained Seq2Seq Baseline weights successfully.")

    def translate_with_attention(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "error": "Empty input text provided",
                "source_text": "",
                "source_tokens": [],
                "target_text": "",
                "target_tokens": [],
                "attention_matrix": [],
                "latency_ms": 0.0
            }

        start_time = time.time()
        src_tokens = tokenize_english(text)
        if not src_tokens:
            return {"error": "Could not extract tokens from input"}

        src_indices = self.src_vocab.sentence_to_indices(src_tokens, add_sos_eos=False)
        # Note: model's translate takes src_indices
        pred_indices, attn_matrix = self.model_attention.translate(
            src_indices,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            max_len=25,
            device=self.device
        )
        latency_ms = (time.time() - start_time) * 1000.0

        target_tokens = self.trg_vocab.indices_to_tokens(pred_indices, remove_special=True)
        target_text = " ".join(target_tokens)

        # Slice attention matrix to match actual target length
        clean_matrix = []
        for i in range(len(target_tokens)):
            if i < len(attn_matrix):
                # Round weights for clear rendering
                row = [round(float(w), 4) for w in attn_matrix[i]]
                clean_matrix.append(row)

        # Alignments & Analysis
        alignments = []
        for i, trg_word in enumerate(target_tokens):
            if i < len(clean_matrix):
                row = clean_matrix[i]
                max_idx = int(row.index(max(row))) if row else 0
                attended_src = src_tokens[max_idx] if max_idx < len(src_tokens) else "N/A"
                confidence = max(row) if row else 0.0
                
                # Compute attention entropy: H = -sum(p * log2(p))
                entropy = 0.0
                for p in row:
                    if p > 1e-7:
                        entropy -= p * math.log2(p)

                alignments.append({
                    "target_index": i,
                    "target_token": trg_word,
                    "primary_source_token": attended_src,
                    "primary_source_index": max_idx,
                    "confidence_score": round(confidence * 100, 1),
                    "entropy": round(entropy, 2),
                    "distribution": [
                        {"source_token": src_tokens[j] if j < len(src_tokens) else f"token_{j}", "weight": round(w * 100, 1)}
                        for j, w in enumerate(row)
                    ]
                })

        self.log_translation(text, target_text, latency_ms, "Bahdanau Attention")

        return {
            "source_text": text,
            "source_tokens": src_tokens,
            "target_text": target_text,
            "target_tokens": target_tokens,
            "attention_matrix": clean_matrix,
            "alignments": alignments,
            "latency_ms": round(latency_ms, 2)
        }

    def translate_baseline(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {"error": "Empty input"}

        start_time = time.time()
        src_tokens = tokenize_english(text)
        src_indices = self.src_vocab.sentence_to_indices(src_tokens, add_sos_eos=False)

        pred_indices = self.model_baseline.translate(
            src_indices,
            sos_idx=SOS_IDX,
            eos_idx=EOS_IDX,
            max_len=25,
            device=self.device
        )
        latency_ms = (time.time() - start_time) * 1000.0

        target_tokens = self.trg_vocab.indices_to_tokens(pred_indices, remove_special=True)
        target_text = " ".join(target_tokens)

        return {
            "source_text": text,
            "source_tokens": src_tokens,
            "target_text": target_text,
            "target_tokens": target_tokens,
            "latency_ms": round(latency_ms, 2)
        }

    def compare_models(self, text: str) -> Dict[str, Any]:
        attn_result = self.translate_with_attention(text)
        base_result = self.translate_baseline(text)

        # Get validation BLEU from training history
        val_bleu_attn = 0.0
        val_bleu_base = 0.0
        if self.training_history:
            attn_bleus = self.training_history.get("attention_model", {}).get("val_bleu", [0.0])
            base_bleus = self.training_history.get("baseline_model", {}).get("val_bleu", [0.0])
            val_bleu_attn = attn_bleus[-1] if attn_bleus else 0.0
            val_bleu_base = base_bleus[-1] if base_bleus else 0.0

        comparison = {
            "input_text": text,
            "model_attention": {
                "name": "Seq2Seq with Bahdanau Attention (Model B)",
                "translation": attn_result.get("target_text", ""),
                "target_tokens": attn_result.get("target_tokens", []),
                "latency_ms": attn_result.get("latency_ms", 0.0),
                "val_bleu_score": val_bleu_attn,
                "attention_matrix": attn_result.get("attention_matrix", []),
                "source_tokens": attn_result.get("source_tokens", []),
                "alignments": attn_result.get("alignments", []),
                "architecture_highlight": "Dynamic context vector c_t computed at every decoding step directly addressing all encoder hidden states."
            },
            "model_baseline": {
                "name": "Seq2Seq Baseline without Attention (Model A)",
                "translation": base_result.get("target_text", ""),
                "target_tokens": base_result.get("target_tokens", []),
                "latency_ms": base_result.get("latency_ms", 0.0),
                "val_bleu_score": val_bleu_base,
                "architecture_highlight": "Fixed-dimension bottleneck vector compressing entire input sequence regardless of length."
            },
            "performance_gain_bleu": round(val_bleu_attn - val_bleu_base, 2)
        }
        return comparison

    def explain_word_attention(self, target_index: int, attention_matrix: List[List[float]], source_tokens: List[str], target_tokens: List[str]) -> Dict[str, Any]:
        if target_index < 0 or target_index >= len(target_tokens) or target_index >= len(attention_matrix):
            return {"error": "Invalid target word index"}

        current_trg_word = target_tokens[target_index]
        row = attention_matrix[target_index]

        # Top attended words
        ranked = sorted(
            [{"token": source_tokens[i] if i < len(source_tokens) else f"tok_{i}", "index": i, "weight": w, "percent": round(w * 100, 1)}
             for i, w in enumerate(row)],
            key=lambda x: x["weight"],
            reverse=True
        )

        top_focus = ranked[0] if ranked else None
        
        # Calculate entropy
        entropy = 0.0
        for p in row:
            if p > 1e-7:
                entropy -= p * math.log2(p)

        # Build Professor-grade explanation in English and Marathi
        if top_focus:
            explanation_en = (
                f"When generating the Marathi word '{current_trg_word}', the decoder directed {top_focus['percent']}% "
                f"of its attention mass to the English token '{top_focus['token']}' (Position {top_focus['index'] + 1}). "
                f"Attention Entropy is {round(entropy, 2)} bits, signifying a "
                f"{'highly peaked, confident' if entropy < 1.5 else 'distributed contextual'} alignment."
            )
            explanation_mr = (
                f"'{current_trg_word}' हा मराठी शब्द तयार करताना मॉडेलने सर्वात जास्त म्हणजेच {top_focus['percent']}% "
                f"लक्ष (Attention) '{top_focus['token']}' या इंग्रजी शब्दावर केंद्रित केले. "
                f"यावरून स्पष्ट होते की AI ने हा शब्द निवडण्यासाठी थेट मूळ भाषेतील '{top_focus['token']}' चा संदर्भ घेतला आहे."
            )
        else:
            explanation_en = "Attention weights not available for this position."
            explanation_mr = "या शब्दासाठी अटेंशन माहिती उपलब्ध नाही."

        return {
            "target_token": current_trg_word,
            "target_index": target_index,
            "ranked_attention": ranked,
            "top_focus": top_focus,
            "entropy": round(entropy, 2),
            "explanation_en": explanation_en,
            "explanation_mr": explanation_mr
        }

    def get_model_info(self) -> Dict[str, Any]:
        param_attn = sum(p.numel() for p in self.model_attention.parameters()) if self.model_attention else 0
        param_base = sum(p.numel() for p in self.model_baseline.parameters()) if self.model_baseline else 0

        return {
            "project_name": "ATTENTION X",
            "tagline": "See How AI Focuses",
            "language_pair": "English to Marathi (en-mr)",
            "attention_mechanism": "Bahdanau (Additive) Attention",
            "models": {
                "attention_model": {
                    "name": "Seq2Seq with Bahdanau Attention",
                    "type": "BiLSTM Encoder + Bahdanau Additive Attention + LSTM Decoder",
                    "parameters": param_attn,
                    "encoder_layers": 1,
                    "encoder_directions": 2,
                    "decoder_layers": 1,
                    "embedding_dim": self.config.get("emb_dim", 128),
                    "hidden_dim": self.config.get("hidden_dim", 256),
                    "attention_dim": self.config.get("attn_dim", 128)
                },
                "baseline_model": {
                    "name": "Seq2Seq Baseline (No Attention)",
                    "type": "LSTM Encoder + LSTM Decoder (Fixed Vector Bottleneck)",
                    "parameters": param_base,
                    "encoder_layers": 1,
                    "encoder_directions": 1,
                    "decoder_layers": 1,
                    "embedding_dim": self.config.get("emb_dim", 128),
                    "hidden_dim": self.config.get("hidden_dim", 256)
                }
            },
            "dataset_info": {
                "total_raw_pairs": self.dataset_meta.get("total_raw_corpus", 51442),
                "active_subset_pairs": self.dataset_meta.get("selected_pairs", 12000),
                "train_pairs": self.dataset_meta.get("train_pairs_count", 10800),
                "val_pairs": self.dataset_meta.get("val_pairs_count", 1200),
                "source_vocab_size": self.src_vocab.n_words if self.src_vocab else 3544,
                "target_vocab_size": self.trg_vocab.n_words if self.trg_vocab else 6000,
                "dataset_source": "Tatoeba English-Marathi Parallel Corpus"
            },
            "training_status": "Ready & Deployed",
            "device": str(self.device)
        }

    def get_analytics(self) -> Dict[str, Any]:
        logs = {"total_translations": 0, "total_latency_ms": 0.0, "history": []}
        if os.path.exists(LOGS_FILE):
            try:
                with open(LOGS_FILE, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                pass

        total_trans = logs.get("total_translations", 0)
        tot_lat = logs.get("total_latency_ms", 0.0)
        avg_latency = round(tot_lat / max(total_trans, 1), 2)

        history = self.training_history
        if not history and os.path.exists(os.path.join(self.models_dir, "training_history.json")):
            with open(os.path.join(self.models_dir, "training_history.json"), "r", encoding="utf-8") as f:
                history = json.load(f)

        return {
            "total_translations": total_trans,
            "average_latency_ms": avg_latency,
            "recent_translations": logs.get("history", [])[-15:],
            "training_history": history,
            "model_summary": self.get_model_info()
        }


# Global engine singleton
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine
