# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# CSP CST 627 100  Deep Learning Neural Networks
# Week 02 Assignment: Attention Under the Hood
# Objective: Hand-Built Self-Attention with Attention-Map and Logit-Lens Probes
"""
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
Attention is all you need. Advances in Neural Information Processing Systems, 30.
https://arxiv.org/abs/1706.03762

Alammar, J. (2018, June 27). The Illustrated Transformer. jalammar.github.io.
https://jalammar.github.io/illustrated-transformer/

Rush, A. (2018). The Annotated Transformer. Harvard NLP.
https://nlp.seas.harvard.edu/annotated-transformer/

nostalgebraist. (2020, August 31). interpreting GPT: the logit lens. LessWrong.
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

# =============================================================================
# endregion
# =============================================================================
# Step 1: Environment Setup
# Get necessary libraries for building and probing a tiny Transformer.
# =============================================================================
"""
https://docs.pytorch.org/docs/stable/index.html
https://docs.pytorch.org/docs/stable/nn.html
https://docs.pytorch.org/docs/stable/nn.functional.html
https://docs.pytorch.org/docs/stable/notes/cuda.html
https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_available.html
https://numpy.org/doc/stable/
https://matplotlib.org/stable/index.html
https://seaborn.pydata.org/
https://docs.python.org/3/library/math.html
https://docs.python.org/3/library/pathlib.html
"""

# Install required libraries (run once in your terminal, not inside the .py script):
# pip install torch numpy matplotlib seaborn

# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Single-source constants
D_MODEL = 64
NUM_HEADS = 4
D_FF = 256
NUM_LAYERS = 2
VOCAB_SIZE = 20
SEQ_LEN = 10
BATCH_SIZE = 32
TRAIN_STEPS = 1000
LEARNING_RATE = 1e-3

# Use NVIDIA CUDA when present (e.g. Colab T4), otherwise CPU (e.g. local VM).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_NAME = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu"
print(f"Using device: {DEVICE} ({DEVICE_NAME})")

# Output directory setup (repo-level results/, regardless of where the script is run from)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# endregion
# =============================================================================
# Step 2: Core Attention Mechanics
# Hand-build scaled dot-product attention and multi-head attention.
# =============================================================================
"""
https://arxiv.org/abs/1706.03762
https://jalammar.github.io/illustrated-transformer/
https://nlp.seas.harvard.edu/annotated-transformer/
https://docs.pytorch.org/docs/stable/generated/torch.matmul.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
"""


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        x, attn_weights = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(x), attn_weights


# =============================================================================
# Step 3: Positional Encodings
# Compare fixed sinusoidal encodings against learned position embeddings.
# =============================================================================
"""
https://arxiv.org/abs/1706.03762
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
https://docs.pytorch.org/docs/stable/generated/torch.arange.html
"""


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


# =============================================================================
# Step 4: Synthetic Task Generator
# Build a copy task so attention behavior is easy to interpret.
# =============================================================================
"""
https://arxiv.org/abs/1410.5401
https://docs.pytorch.org/docs/stable/generated/torch.randint.html
https://docs.pytorch.org/docs/stable/generated/torch.cat.html
https://docs.pytorch.org/docs/stable/generated/torch.zeros.html
"""


def generate_copy_task_batch(batch_size, seq_len, vocab_size):
    """
    Task: Read a sequence, encounter a separator token (0), and output the exact sequence.
    Vocab uses tokens 1 to vocab_size-1. Token 0 is the separator.
    """
    # Generated on CPU so the seeded data stream is identical on CPU and CUDA
    # runs, then moved to the active device.
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    sep = torch.zeros(batch_size, 1, dtype=torch.long)
    # .contiguous() so the slices support .view() later in the loss.
    inputs = torch.cat([x, sep, x], dim=1)[:, :-1].contiguous()
    targets = torch.cat([x, sep, x], dim=1)[:, 1:].contiguous()
    return inputs.to(DEVICE), targets.to(DEVICE)


# =============================================================================
# Step 5: Tiny Transformer Construction & Training Loop
# Stack attention + feed-forward blocks and train on the copy task.
# =============================================================================
"""
https://arxiv.org/abs/1706.03762
https://arxiv.org/abs/1607.06450
https://nlp.seas.harvard.edu/annotated-transformer/
https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
"""


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x, attn_weights


class TinyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, pos_enc_type="learned"
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

        if pos_enc_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model)
        else:
            self.pos_encoder = LearnedPositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_intermediates=False):
        hidden_states = []
        attention_maps = []

        x = self.embedding(x)
        x = self.pos_encoder(x)
        hidden_states.append(x)

        for layer in self.layers:
            x, attn_weights = layer(x)
            attention_maps.append(attn_weights)
            hidden_states.append(x)

        logits = self.unembed(x)

        if return_intermediates:
            return logits, attention_maps, hidden_states
        return logits


def train_model(model):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"Training model on copy task ({DEVICE})...")
    for step in range(TRAIN_STEPS):
        inputs, targets = generate_copy_task_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    print("Training complete.")
    return model


# =============================================================================
# Step 6: Interpretability Probes (Attention Maps & Logit Lens)
# Visualize per-head attention weights and decode intermediate hidden states.
# =============================================================================
"""
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
https://arxiv.org/abs/1906.05714
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.table.html
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
"""


def plot_attention_maps(attention_maps, tokens, seq_idx, filename="attention_maps.png"):
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].size(1)

    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(3 * num_heads, 3 * num_layers)
    )

    token_str = [str(t.item()) for t in tokens[seq_idx]]

    for l in range(num_layers):
        for h in range(num_heads):
            ax = axes[l, h] if num_layers > 1 else axes[h]
            sns.heatmap(
                # .cpu() so CUDA tensors can be handed to NumPy/seaborn.
                attention_maps[l][seq_idx, h].detach().cpu().numpy(),
                xticklabels=token_str,
                yticklabels=token_str,
                cmap="viridis",
                cbar=False,
                ax=ax,
            )
            ax.set_title(f"L{l}H{h}")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Attention map saved to {OUTPUT_DIR / filename}")


def apply_logit_lens(model, hidden_states, tokens, seq_idx, filename="logit_lens.png"):
    num_layers = len(hidden_states)
    seq_length = hidden_states[0].size(1)

    token_str = [str(t.item()) for t in tokens[seq_idx]]
    top_predictions = []

    for l in range(num_layers):
        h = hidden_states[l][seq_idx]
        logits = model.unembed(h)
        preds = torch.argmax(logits, dim=-1)
        top_predictions.append([str(p.item()) for p in preds])

    fig, ax = plt.subplots(figsize=(10, num_layers * 0.8))
    ax.axis("off")
    ax.axis("tight")

    table_data = []
    for l in range(num_layers):
        table_data.append([f"Layer {l}"] + top_predictions[l])

    cols = ["Layer"] + token_str
    table = ax.table(
        cellText=table_data, colLabels=cols, loc="center", cellLoc="center"
    )
    table.scale(1, 2)

    plt.title("Logit Lens: Intermediate Predictions")
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()
    print(f"Logit lens saved to {OUTPUT_DIR / filename}")


def make_probe_input(prefix):
    """Build a copy-task input from a hand-picked prefix (same layout as the generator)."""
    x = torch.tensor(prefix, dtype=torch.long).unsqueeze(0)
    sep = torch.zeros(1, 1, dtype=torch.long)
    return torch.cat([x, sep, x], dim=1)[:, :-1].to(DEVICE)


# Hand-picked prefixes: repeats test position vs token identity, ascending is a
# repeat-free baseline, constant removes all token-identity signal.
HAND_PICKED_PREFIXES = {
    "repeats": [5, 3, 9, 3, 7, 5, 2, 8, 2, 6],
    "ascending": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "constant": [7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
}


def run_probes(model):
    model.eval()
    seeded, _ = generate_copy_task_batch(1, SEQ_LEN, VOCAB_SIZE)
    probe_inputs = [("", seeded)] + [
        (f"_{name}", make_probe_input(prefix))
        for name, prefix in HAND_PICKED_PREFIXES.items()
    ]

    for name, inputs in probe_inputs:
        with torch.no_grad():
            logits, attention_maps, hidden_states = model(
                inputs, return_intermediates=True
            )

        plot_attention_maps(
            attention_maps, inputs, 0, filename=f"attention_maps{name}.png"
        )
        apply_logit_lens(
            model, hidden_states, inputs, 0, filename=f"logit_lens{name}.png"
        )


# =============================================================================
# Step 7: Run the Experiment
# Seed for reproducibility, train the model, and generate probe outputs.
# =============================================================================
"""
https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html
https://docs.pytorch.org/docs/stable/notes/randomness.html
"""

if __name__ == "__main__":
    torch.manual_seed(42)
    model = TinyTransformer(
        VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, pos_enc_type="learned"
    )
    model = train_model(model)
    run_probes(model)
