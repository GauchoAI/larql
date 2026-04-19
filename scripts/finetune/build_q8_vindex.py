#!/usr/bin/env python3
"""Build Q8_0 interleaved FFN weights + attention weights from fused safetensors.

Creates files compatible with larql's Metal pipeline:
- interleaved_q4k_real.bin → Q8_0 format (34 bytes per 32 values)
  Named q4k for compatibility but actually Q8_0 data
- attn_weights_q4k.bin + manifest → Q8_0 attention weights

The Q8_0 format preserves LoRA fine-tuning signal that Q4_K loses.
"""

import json
import struct
import numpy as np
from pathlib import Path
from safetensors import safe_open

def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize f32 values to GGUF Q8_0 format (34 bytes per 32 values)."""
    data = data.astype(np.float32).flatten()
    n = len(data)
    assert n % 32 == 0, f"length must be multiple of 32, got {n}"

    blocks = n // 32
    result = bytearray()

    for b in range(blocks):
        block = data[b * 32 : (b + 1) * 32]
        # Scale = max absolute value
        amax = np.max(np.abs(block))
        if amax == 0:
            d = np.float16(0.0)
            quants = np.zeros(32, dtype=np.int8)
        else:
            d = np.float16(amax / 127.0)
            scale = float(d)
            if scale == 0:
                quants = np.zeros(32, dtype=np.int8)
            else:
                quants = np.clip(np.round(block / scale), -128, 127).astype(np.int8)

        # Pack: f16 scale (2 bytes) + 32 int8 quants (32 bytes) = 34 bytes
        result += struct.pack('<e', float(d))
        result += quants.tobytes()

    return bytes(result)

def main():
    import sys

    fused_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scripts/finetune/fused")
    vindex_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "Desktop/llm-as-a-database/gemma3-4b-ft.vindex"

    print(f"=== Build Q8_0 weights for larql vindex ===")
    print(f"  Source: {fused_dir}")
    print(f"  Target: {vindex_dir}")

    # Load config
    with open(fused_dir / "config.json") as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    num_layers = text_config.get("num_hidden_layers", 34)
    hidden = text_config.get("hidden_size", 2560)
    inter = text_config.get("intermediate_size", 10240)
    num_q_heads = text_config.get("num_attention_heads", 8)
    num_kv_heads = text_config.get("num_key_value_heads", 4)
    head_dim = text_config.get("head_dim", hidden // num_q_heads)

    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"  Layers: {num_layers}, hidden: {hidden}, inter: {inter}")
    print(f"  Q heads: {num_q_heads}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
    print(f"  Q dim: {q_dim}, KV dim: {kv_dim}")

    # Open safetensors
    shard_files = sorted(fused_dir.glob("model-*.safetensors"))
    if not shard_files:
        shard_files = sorted(fused_dir.glob("*.safetensors"))

    print(f"  Shards: {len(shard_files)}")

    # Build tensor name → shard index mapping
    tensor_map = {}
    shards = []
    for sf in shard_files:
        st = safe_open(str(sf), framework="numpy")
        shards.append(st)
        for key in st.keys():
            tensor_map[key] = len(shards) - 1

    def get_tensor(name):
        """Get tensor by name, trying various key formats. Converts bf16 → f32."""
        for prefix in ["model.layers.", "language_model.model.layers.", ""]:
            key = f"{prefix}{name}"
            if key in tensor_map:
                try:
                    return shards[tensor_map[key]].get_tensor(key).astype(np.float32)
                except TypeError:
                    # BF16 not understood by numpy — read raw bytes and convert
                    st = shards[tensor_map[key]]
                    # Use torch for bf16 conversion
                    import torch
                    with safe_open(str(shard_files[tensor_map[key]]), framework="pt") as f:
                        return f.get_tensor(key).float().numpy()
        raise KeyError(f"tensor not found: {name}")

    # ── Build interleaved Q8_0 FFN weights ──
    print(f"\n  Building interleaved FFN weights (Q8_0)...")

    ffn_out = open(vindex_dir / "interleaved_q4k_real.bin", "wb")
    bytes_per_matrix = (inter * hidden) // 32 * 34  # Q8_0: 34 bytes per 32 values

    for layer in range(num_layers):
        gate = get_tensor(f"{layer}.mlp.gate_proj.weight")
        up = get_tensor(f"{layer}.mlp.up_proj.weight")
        down = get_tensor(f"{layer}.mlp.down_proj.weight")

        gate_q8 = quantize_q8_0(gate)
        up_q8 = quantize_q8_0(up)
        down_q8 = quantize_q8_0(down)

        ffn_out.write(gate_q8)
        ffn_out.write(up_q8)
        ffn_out.write(down_q8)

        if layer % 10 == 0 or layer == num_layers - 1:
            print(f"    Layer {layer}/{num_layers}: gate={len(gate_q8)}, up={len(up_q8)}, down={len(down_q8)}")

    ffn_out.close()
    ffn_size = (vindex_dir / "interleaved_q4k_real.bin").stat().st_size
    print(f"  FFN written: {ffn_size / 1e6:.1f} MB")

    # ── Build Q8_0 attention weights ──
    print(f"\n  Building attention weights (Q8_0)...")

    attn_out = open(vindex_dir / "attn_weights_q4k.bin", "wb")
    manifest = []

    for layer in range(num_layers):
        for proj_name, proj_key, dim in [
            ("q", f"{layer}.self_attn.q_proj.weight", q_dim),
            ("k", f"{layer}.self_attn.k_proj.weight", kv_dim),
            ("v", f"{layer}.self_attn.v_proj.weight", kv_dim),
            ("o", f"{layer}.self_attn.o_proj.weight", hidden),
        ]:
            w = get_tensor(proj_key)
            q8 = quantize_q8_0(w)
            offset = attn_out.tell()
            attn_out.write(q8)
            manifest.append({
                "offset": offset,
                "length": len(q8),
                "format": "Q8_0",
                "rows": dim,
                "cols": hidden if proj_name != "o" else q_dim,
            })

        if layer % 10 == 0 or layer == num_layers - 1:
            print(f"    Layer {layer}/{num_layers}")

    attn_out.close()
    attn_size = (vindex_dir / "attn_weights_q4k.bin").stat().st_size
    print(f"  Attention written: {attn_size / 1e6:.1f} MB")

    # Write manifest
    with open(vindex_dir / "attn_weights_q4k_manifest.json", "w") as f:
        json.dump(manifest, f)

    # ── Build embeddings ──
    print(f"\n  Building embeddings...")
    embed_key = None
    for k in tensor_map:
        if "embed_tokens" in k:
            embed_key = k
            break

    if embed_key:
        try:
            embed = shards[tensor_map[embed_key]].get_tensor(embed_key).astype(np.float32)
        except TypeError:
            import torch
            with safe_open(str(shard_files[tensor_map[embed_key]]), framework="pt") as f:
                embed = f.get_tensor(embed_key).float().numpy()
        embed.tofile(str(vindex_dir / "embeddings.bin"))
        print(f"  Embeddings: {embed.shape} → {embed.nbytes / 1e6:.1f} MB")

    # ── Build norms ──
    print(f"\n  Building norms...")
    norms_data = bytearray()
    norm_keys_found = 0

    for key in sorted(tensor_map.keys()):
        if "norm" in key.lower() or "layernorm" in key.lower():
            try:
                w = shards[tensor_map[key]].get_tensor(key).astype(np.float32)
            except TypeError:
                import torch
                with safe_open(str(shard_files[tensor_map[key]]), framework="pt") as f:
                    w = f.get_tensor(key).float().numpy()
            norms_data += w.tobytes()
            norm_keys_found += 1

    if norm_keys_found > 0:
        with open(vindex_dir / "norms.bin", "wb") as f:
            f.write(norms_data)
        print(f"  Norms: {norm_keys_found} vectors, {len(norms_data) / 1e3:.1f} KB")

    # ── Build lm_head ──
    print(f"\n  Building lm_head...")
    lm_head_key = None
    for k in tensor_map:
        if "lm_head" in k:
            lm_head_key = k
            break

    if lm_head_key:
        try:
            lm_head = shards[tensor_map[lm_head_key]].get_tensor(lm_head_key).astype(np.float32)
        except TypeError:
            import torch
            with safe_open(str(shard_files[tensor_map[lm_head_key]]), framework="pt") as f:
                lm_head = f.get_tensor(lm_head_key).float().numpy()
        lm_head_q8 = quantize_q8_0(lm_head)
        with open(vindex_dir / "lm_head_q4.bin", "wb") as f:
            f.write(lm_head_q8)
        print(f"  lm_head: {lm_head.shape} → Q8_0 {len(lm_head_q8) / 1e6:.1f} MB")
    else:
        # Tied to embeddings
        print(f"  lm_head: tied to embeddings (no separate file)")

    print(f"\n=== Done ===")
    total = sum(f.stat().st_size for f in vindex_dir.glob("*.bin") if not f.is_symlink())
    print(f"  Total weight data: {total / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
