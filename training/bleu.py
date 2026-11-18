import math
from collections import Counter
from typing import List

def get_ngrams(tokens: List[str], n: int) -> Counter:
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

def calculate_sentence_bleu(reference: List[str], candidate: List[str], max_n: int = 4, weights: List[float] = None) -> float:
    if not candidate or not reference:
        return 0.0

    if weights is None:
        weights = [1.0 / max_n] * max_n

    c = len(candidate)
    r = len(reference)

    # Brevity penalty
    if c == 0:
        return 0.0
    if c > r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(r) / float(c))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = get_ngrams(reference, n)
        cand_ngrams = get_ngrams(candidate, n)

        total_cand = sum(cand_ngrams.values())
        if total_cand == 0:
            precisions.append(0.0)
            continue

        # Clipped counts
        overlap = 0
        for ngram, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))

        precisions.append(overlap / total_cand)

    # Smoothing (add epsilon if precision is 0 for higher n-grams)
    log_precisions = []
    for p in precisions:
        if p > 0:
            log_precisions.append(math.log(p))
        else:
            log_precisions.append(math.log(1e-6))

    score = bp * math.exp(sum(w * lp for w, lp in zip(weights, log_precisions)))
    return round(score * 100.0, 2)


def calculate_corpus_bleu(references: List[List[str]], candidates: List[List[str]]) -> float:
    scores = [calculate_sentence_bleu(ref, cand) for ref, cand in zip(references, candidates)]
    return round(sum(scores) / max(len(scores), 1), 2)
