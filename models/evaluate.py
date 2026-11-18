import os
import json
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.inference import get_engine
from training.bleu import calculate_sentence_bleu, calculate_corpus_bleu

def run_evaluation(num_samples: int = 20):
    print("=" * 60)
    print("ATTENTION X • Neural Translation Model Evaluation")
    print("=" * 60)

    engine = get_engine()
    
    val_pairs_path = "dataset/val_pairs.json"
    if not os.path.exists(val_pairs_path):
        print("Validation dataset not found!")
        return

    with open(val_pairs_path, "r", encoding="utf-8") as f:
        val_pairs = json.load(f)

    test_samples = val_pairs[:num_samples]
    print(f"Evaluating {len(test_samples)} validation sentence pairs...\n")

    attn_refs = []
    attn_cands = []
    base_refs = []
    base_cands = []

    print(f"{'Input (English)':<30} | {'Model B (Bahdanau)':<30} | {'Model A (Baseline)':<30}")
    print("-" * 96)

    for pair in test_samples:
        src_text = pair["src_raw"]
        ref_tokens = pair["trg_tokens"]

        # Run Model B (Attention)
        attn_res = engine.translate_with_attention(src_text)
        attn_pred_tokens = attn_res.get("target_tokens", [])
        attn_refs.append(ref_tokens)
        attn_cands.append(attn_pred_tokens)

        # Run Model A (Baseline)
        base_res = engine.translate_baseline(src_text)
        base_pred_tokens = base_res.get("target_tokens", [])
        base_refs.append(ref_tokens)
        base_cands.append(base_pred_tokens)

        src_disp = src_text[:28]
        attn_disp = attn_res.get("target_text", "")[:28]
        base_disp = base_res.get("target_text", "")[:28]
        print(f"{src_disp:<30} | {attn_disp:<30} | {base_disp:<30}")

    bleu_attn = calculate_corpus_bleu(attn_refs, attn_cands)
    bleu_base = calculate_corpus_bleu(base_refs, base_cands)

    print("-" * 96)
    print(f"\nModel B (Bahdanau Attention) Corpus BLEU: {bleu_attn:.2f}")
    print(f"Model A (Baseline No Attention) Corpus BLEU: {bleu_base:.2f}")
    print(f"Attention Performance Gain: +{bleu_attn - bleu_base:.2f} BLEU points")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
