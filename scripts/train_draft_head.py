#!/usr/bin/env python3
"""
Train a draft head MLP for speculative decoding.

Input: binary file of (h_final[2560], token_id) pairs
Output: draft_head.bin — two weight matrices for Metal inference

Architecture: h[2560] → Linear(2560, 1024) → GELU → Linear(1024, vocab)
Small enough to run in ~0.1ms on Metal (one Q4_K-sized matvec).

Usage:
  python3 scripts/train_draft_head.py /tmp/draft_train.bin draft_head.bin
"""

import sys
import struct
import numpy as np

# Use MLX (Apple Silicon native) or PyTorch
try:
    import mlx.core as mx
    import mlx.nn as mnn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_MLX and not HAS_TORCH:
    print("Need either MLX or PyTorch for training")

HIDDEN = 2560
INNER = 1024
RECORD_SIZE = HIDDEN * 4 + 4  # f32 × 2560 + u32

def load_data(path):
    """Load (h_final, token_id) pairs from binary file."""
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // RECORD_SIZE
    print(f"Loading {n} training records from {path}")

    h_all = np.zeros((n, HIDDEN), dtype=np.float32)
    tokens = np.zeros(n, dtype=np.int64)

    for i in range(n):
        offset = i * RECORD_SIZE
        h_bytes = data[offset:offset + HIDDEN * 4]
        tok_bytes = data[offset + HIDDEN * 4:offset + RECORD_SIZE]
        h_all[i] = np.frombuffer(h_bytes, dtype=np.float32)
        tokens[i] = struct.unpack("<I", tok_bytes)[0]

    return h_all, tokens

def train_mlx(h_all, tokens, vocab_size, epochs=10, lr=1e-3):
    """Train with MLX (Apple Silicon native)."""
    print("Training with MLX on Apple Silicon GPU")

    class DraftHead(mnn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = mnn.Linear(HIDDEN, INNER)
            self.fc2 = mnn.Linear(INNER, vocab_size)

        def __call__(self, x):
            return self.fc2(mnn.gelu(self.fc1(x)))

    model = DraftHead()
    mx.eval(model.parameters())

    optimizer = optim.AdamW(learning_rate=lr)

    X = mx.array(h_all)
    Y = mx.array(tokens.astype(np.int32))
    n = len(h_all)
    batch_size = 256

    def loss_fn(model, xb, yb):
        logits = model(xb)
        return mx.mean(mnn.losses.cross_entropy(logits, yb))

    loss_and_grad = mnn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        correct = 0
        batches = 0

        for i in range(0, n, batch_size):
            idx = mx.array(perm[i:i+batch_size])
            xb = X[idx]
            yb = Y[idx]

            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            preds = model(xb).argmax(axis=1)
            correct += (preds == yb).sum().item()
            batches += 1

        acc = correct / n * 100
        avg_loss = total_loss / batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.1f}%")

    # Extract weights as numpy
    params = model.parameters()
    w1 = np.array(model.fc1.weight)   # [INNER, HIDDEN]
    b1 = np.array(model.fc1.bias)     # [INNER]
    w2 = np.array(model.fc2.weight)   # [vocab, INNER]
    b2 = np.array(model.fc2.bias)     # [vocab]

    # Top-5 accuracy
    logits = model(X)
    top1 = logits.argmax(axis=1)
    top1_acc = (top1 == Y).sum().item() / n * 100
    print(f"  Final top-1 accuracy: {top1_acc:.1f}%")

    return w1, b1, w2, b2

def train_torch(h_all, tokens, vocab_size, epochs=10, lr=1e-3):
    """Train with PyTorch."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    class DraftHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(HIDDEN, INNER)
            self.fc2 = nn.Linear(INNER, vocab_size)

        def forward(self, x):
            x = torch.nn.functional.gelu(self.fc1(x))
            return self.fc2(x)

    model = DraftHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(h_all, device=device)
    Y = torch.tensor(tokens, device=device)

    n = len(h_all)
    batch_size = 256

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        correct = 0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X[idx]
            yb = Y[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            batches += 1

        acc = correct / n * 100
        avg_loss = total_loss / batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.1f}%")

    # Extract weights
    w1 = model.fc1.weight.detach().cpu().numpy()  # [INNER, HIDDEN]
    b1 = model.fc1.bias.detach().cpu().numpy()     # [INNER]
    w2 = model.fc2.weight.detach().cpu().numpy()  # [vocab, INNER]
    b2 = model.fc2.bias.detach().cpu().numpy()     # [vocab]

    # Top-5 accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X)
        top5 = logits.topk(5, dim=1).indices
        top5_correct = sum(1 for i in range(n) if Y[i] in top5[i])
    print(f"  Top-1 accuracy: {acc:.1f}%")
    print(f"  Top-5 accuracy: {top5_correct/n*100:.1f}%")

    return w1, b1, w2, b2

def save_weights(path, w1, b1, w2, b2):
    """Save as binary: w1[INNER×HIDDEN] + b1[INNER] + w2[vocab×INNER] + b2[vocab]."""
    with open(path, "wb") as f:
        # Header: magic + dims
        f.write(b"DRFT")  # magic
        f.write(struct.pack("<IIII", HIDDEN, INNER, w2.shape[0], 0))  # hidden, inner, vocab, reserved
        # Weights as f32
        f.write(w1.astype(np.float32).tobytes())
        f.write(b1.astype(np.float32).tobytes())
        f.write(w2.astype(np.float32).tobytes())
        f.write(b2.astype(np.float32).tobytes())

    size_mb = (w1.nbytes + b1.nbytes + w2.nbytes + b2.nbytes) / 1e6
    print(f"Saved {path}: {size_mb:.1f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <train_data.bin> <output_weights.bin>")
        sys.exit(1)

    train_path = sys.argv[1]
    output_path = sys.argv[2]

    h_all, tokens = load_data(train_path)

    # Vocab size: max token ID + 1
    vocab_size = int(tokens.max()) + 1
    # Cap at reasonable size (Gemma 3 has 262208)
    # For the draft head, we only need to predict tokens that actually appear
    # But the full vocab is needed for correct token IDs
    print(f"Vocab size: {vocab_size}")
    print(f"Unique tokens in training: {len(np.unique(tokens))}")

    if HAS_MLX:
        w1, b1, w2, b2 = train_mlx(h_all, tokens, vocab_size, epochs=20, lr=3e-4)
    elif HAS_TORCH:
        w1, b1, w2, b2 = train_torch(h_all, tokens, vocab_size, epochs=20, lr=3e-4)
    else:
        print("ERROR: Need MLX or PyTorch for training")
        sys.exit(1)

    save_weights(output_path, w1, b1, w2, b2)
