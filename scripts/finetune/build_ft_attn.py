#!/usr/bin/env python3
"""Build Q8_0 attention weights ONLY for LoRA-affected layers.

The LoRA adapter modifies layers 18-33 (last 16 of 34).
Only attention Q/K/V/O projections are changed.
FFN, embeddings, norms, lm_head are all unchanged.

This script builds a hybrid attn_weights_q4k.bin:
- Layers 0-17: Q4_K from original vindex (unchanged)
- Layers 18-33: Q8_0 from fused model (LoRA changes preserved)
"""

import json
import struct
import numpy as np
from pathlib import Path

def quantize_q8_0(data: np.ndarray) -> bytes:
    """Quantize f32 values to GGUF Q8_0 format (34 bytes per 32 values)."""
    data = data.astype(np.float32).flatten()
    n = len(data)
    assert n % 32 == 0, f"length must be multiple of 32, got {n}"
    blocks = n // 32
    result = bytearray()
    for b in range(blocks):
        block = data[b * 32 : (b + 1) * 32]
        amax = np.max(np.abs(block))
        if amax == 0:
            d = np.float16(0.0)
            quants = np.zeros(32, dtype=np.int8)
        else:
            d = np.float16(amax / 127.0)
            scale = float(d)
            quants = np.clip(np.round(block / scale), -128, 127).astype(np.int8) if scale != 0 else np.zeros(32, dtype=np.int8)
        result += struct.pack('<e', float(d))
        result += quants.tobytes()
    return bytes(result)

def main():
    import sys

    fused_dir = Path("scripts/finetune/fused")
    orig_vindex = Path.home() / "Desktop/llm-as-a-database/gemma3-4b.vindex"
    ft_vindex = Path.home() / "Desktop/llm-as-a-database/gemma3-4b-ft.vindex"

    # Config
    lora_start_layer = 18  # layers 18-33 have LoRA
    num_layers = 34

    # Read original manifest
    with open(orig_vindex / "attn_weights_q4k_manifest.json") as f:
        orig_manifest = json.load(f)

    # Read original attn weights (Q4_K)
    orig_data = open(orig_vindex / "attn_weights_q4k.bin", "rb").read()
    print(f"Original attn weights: {len(orig_data) / 1e6:.1f} MB")

    # Load fused safetensors
    from safetensors import safe_open
    import torch

    shard_files = sorted(fused_dir.glob("model-*.safetensors"))
    tensor_map = {}
    for i, sf in enumerate(shard_files):
        with safe_open(str(sf), framework="pt") as st:
            for key in st.keys():
                tensor_map[key] = (i, sf)

    def get_tensor_f32(name):
        for prefix in ["language_model.model.layers.", "model.layers.", ""]:
            key = f"{prefix}{name}"
            if key in tensor_map:
                _, sf = tensor_map[key]
                with safe_open(str(sf), framework="pt") as st:
                    return st.get_tensor(key).float().numpy()
        raise KeyError(f"tensor not found: {name}")

    # Read config for dimensions
    with open(fused_dir / "config.json") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    hidden = text_config.get("hidden_size", 2560)
    num_q_heads = text_config.get("num_attention_heads", 8)
    num_kv_heads = text_config.get("num_key_value_heads", 4)
    head_dim = text_config.get("head_dim", hidden // num_q_heads)
    q_dim = num_q_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    print(f"hidden={hidden}, q_dim={q_dim}, kv_dim={kv_dim}")
    print(f"LoRA layers: {lora_start_layer}-{num_layers-1}")

    # Build hybrid file
    out = open(ft_vindex / "attn_weights_q4k.bin", "wb")
    new_manifest = []

    for layer in range(num_layers):
        entries_per_layer = 4  # Q, K, V, O
        base_idx = layer * entries_per_layer

        if layer < lora_start_layer:
            # Use original Q4_K data (unchanged)
            for i in range(entries_per_layer):
                entry = orig_manifest[base_idx + i]
                chunk = orig_data[entry["offset"]:entry["offset"] + entry["length"]]
                new_offset = out.tell()
                out.write(chunk)
                new_manifest.append({
                    "offset": new_offset,
                    "length": len(chunk),
                    "format": entry["format"],
                })
        else:
            # Build Q8_0 from fused weights
            for proj_name, proj_key, rows in [
                ("q", f"{layer}.self_attn.q_proj.weight", q_dim),
                ("k", f"{layer}.self_attn.k_proj.weight", kv_dim),
                ("v", f"{layer}.self_attn.v_proj.weight", kv_dim),
                ("o", f"{layer}.self_attn.o_proj.weight", hidden),
            ]:
                w = get_tensor_f32(proj_key)
                q8 = quantize_q8_0(w)
                new_offset = out.tell()
                out.write(q8)
                new_manifest.append({
                    "offset": new_offset,
                    "length": len(q8),
                    "format": "Q8_0",
                })

        if layer % 10 == 0 or layer == num_layers - 1:
            fmt = "Q4_K" if layer < lora_start_layer else "Q8_0"
            print(f"  Layer {layer}: {fmt}")

    out.close()

    # Write manifest
    with open(ft_vindex / "attn_weights_q4k_manifest.json", "w") as f:
        json.dump(new_manifest, f)

    total = (ft_vindex / "attn_weights_q4k.bin").stat().st_size
    print(f"\nHybrid attention weights: {total / 1e6:.1f} MB")
    q4k_layers = lora_start_layer
    q8_layers = num_layers - lora_start_layer
    print(f"  Q4_K: {q4k_layers} layers (original)")
    print(f"  Q8_0: {q8_layers} layers (fine-tuned)")

if __name__ == "__main__":
    main()
