import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

# Repo-level results/, regardless of where the script is run from
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Step 1: Core Attention Mechanics
class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v), attn_weights


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
        b = q.size(0)
        q = self.w_q(q).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, -1, self.num_heads, self.d_k).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
        x, attn_weights = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.w_o(x), attn_weights


# Step 2: Positional Encodings
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
        pos = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(pos)


# Step 3: Synthetic Task Generator
def generate_copy_task_batch(batch_size, seq_len, vocab_size):
    # Generated on CPU so the seeded data stream is identical on CPU and CUDA,
    # .contiguous() so the slices support .view() later in the loss.
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    sep = torch.zeros(batch_size, 1, dtype=torch.long)
    inputs = torch.cat([x, sep, x], dim=1)[:, :-1].contiguous()
    targets = torch.cat([x, sep, x], dim=1)[:, 1:].contiguous()
    return inputs.to(DEVICE), targets.to(DEVICE)


# Step 4: Tiny Transformer Construction & Training Loop
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
        x = self.ln2(x + self.ffn(x))
        return x, attn_weights


class TinyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, pos_enc_type="learned"
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = (
            SinusoidalPositionalEncoding(d_model)
            if pos_enc_type == "sinusoidal"
            else LearnedPositionalEncoding(d_model)
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, x, return_intermediates=False):
        hidden_states, attention_maps = [], []
        x = self.pos_encoder(self.embedding(x))
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
    for step in range(TRAIN_STEPS):
        inputs, targets = generate_copy_task_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()
        loss = criterion(model(inputs).view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
    return model


# Step 5: Interpretability Probes (Attention Maps & Logit Lens)
def plot_attention_maps(attention_maps, tokens, seq_idx, filename="attention_maps.png"):
    num_layers, num_heads = len(attention_maps), attention_maps[0].size(1)
    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(3 * num_heads, 3 * num_layers)
    )
    token_str = [str(t.item()) for t in tokens[seq_idx]]
    for l in range(num_layers):
        for h in range(num_heads):
            ax = axes[l, h] if num_layers > 1 else axes[h]
            # .cpu() so CUDA tensors can be handed to NumPy/seaborn.
            sns.heatmap(
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


def apply_logit_lens(model, hidden_states, tokens, seq_idx, filename="logit_lens.png"):
    num_layers = len(hidden_states)
    token_str = [str(t.item()) for t in tokens[seq_idx]]
    top_preds = [
        [
            str(p.item())
            for p in torch.argmax(model.unembed(hidden_states[l][seq_idx]), dim=-1)
        ]
        for l in range(num_layers)
    ]
    fig, ax = plt.subplots(figsize=(10, num_layers * 0.8))
    ax.axis("off")
    ax.axis("tight")
    table_data = [[f"Layer {l}"] + top_preds[l] for l in range(num_layers)]
    ax.table(
        cellText=table_data,
        colLabels=["Layer"] + token_str,
        loc="center",
        cellLoc="center",
    ).scale(1, 2)
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def run_probes(model):
    model.eval()
    inputs, _ = generate_copy_task_batch(1, SEQ_LEN, VOCAB_SIZE)
    with torch.no_grad():
        logits, attention_maps, hidden_states = model(inputs, return_intermediates=True)
    plot_attention_maps(attention_maps, inputs, 0)
    apply_logit_lens(model, hidden_states, inputs, 0)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = train_model(
        TinyTransformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS)
    )
    run_probes(model)
