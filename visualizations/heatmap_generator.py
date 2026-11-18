import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_and_save_heatmap(
    source_tokens: List[str],
    target_tokens: List[str],
    attention_matrix: List[List[float]],
    output_path: str = "visualizations/attention_heatmap.png",
    dpi: int = 200
):
    """
    Generates a publication-grade attention alignment heatmap using Matplotlib.
    Horizontal axis: Source words (English)
    Vertical axis: Target words (Marathi)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    matrix = np.array(attention_matrix)

    # Set dark cyber style matching ATTENTION X design theme
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(max(6, len(source_tokens) * 0.9), max(5, len(target_tokens) * 0.8)), dpi=dpi)
    fig.patch.set_facecolor("#070b14")
    ax.set_facecolor("#0d1424")

    # Heatmap color map: Purple to Cyan
    cmap = "viridis"
    cax = ax.matshow(matrix, cmap=cmap, aspect="auto")

    # Colorbar
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attention Score", color="#f8fafc", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(cbar.ax.get_yticklabels(), color="#94a3b8")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(source_tokens)))
    ax.set_yticks(np.arange(len(target_tokens)))

    ax.set_xticklabels(source_tokens, fontsize=11, rotation=45, ha="left", color="#00f2fe", weight="bold")
    ax.set_yticklabels(target_tokens, fontsize=12, color="#c084fc", weight="bold")

    ax.set_xlabel("Source Words (English)", color="#00f2fe", fontsize=12, labelpad=14, weight="bold")
    ax.set_ylabel("Target Words (Marathi)", color="#c084fc", fontsize=12, labelpad=14, weight="bold")

    ax.set_title("ATTENTION X • Bahdanau Additive Attention Alignment", color="#ffffff", fontsize=13, pad=25, weight="bold")

    # Annotate values inside cells
    for i in range(len(target_tokens)):
        for j in range(len(source_tokens)):
            val = matrix[i, j]
            text_color = "#070b14" if val > 0.5 else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9, weight="bold")

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Attention heatmap saved successfully to {output_path}")
    return output_path
