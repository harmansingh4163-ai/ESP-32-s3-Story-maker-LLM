#!/usr/bin/env python3
"""Generate a small random model in the ESPL format + reference logits
computed by a NumPy forward pass that mirrors llm_core.c exactly
(same quantization, same integer-accumulate matmul)."""
import numpy as np, struct, sys

MAGIC = 0x4C505345

def round_away(x):  # match C roundf (half away from zero)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def quant_weights(w, bits, gs):
    """w: (rows, n) fp32 -> (scales (rows, n/gs) f32, packed bytes)"""
    rows, n = w.shape
    g = w.reshape(rows, n // gs, gs)
    wmax = np.abs(g).max(axis=2)
    qmax = 127.0 if bits == 8 else 7.0
    scale = np.where(wmax == 0, 1.0, wmax / qmax).astype(np.float32)
    q = np.clip(round_away(g / scale[..., None]), -qmax, qmax).astype(np.int32)
    if bits == 8:
        packed = q.astype(np.int8).reshape(rows, n).tobytes()
    else:
        qn = (q + 8).astype(np.uint8).reshape(rows, n)
        packed = (qn[:, 0::2] | (qn[:, 1::2] << 4)).tobytes()
    return scale, packed, q.reshape(rows, n)  # q for the reference

def quant_act(x, gs):
    g = x.reshape(-1, gs)
    m = np.abs(g).max(axis=1)
    scale = np.where(m == 0, 1.0, m / 127.0).astype(np.float32)
    q = round_away(g / scale[:, None]).astype(np.int32)
    return q.reshape(-1), scale

def qmatmul(qw, sw, x, gs):
    """mirror llm_matmul_rows: int32 group dots, fp32 group scaling"""
    qx, sx = quant_act(x.astype(np.float32), gs)
    rows, n = qw.shape
    a = (qw.reshape(rows, n//gs, gs) * qx.reshape(1, n//gs, gs)).sum(2)  # int32 exact
    return ((a.astype(np.float32)) * sw * sx[None, :]).sum(1).astype(np.float32)

def rmsnorm(x, g):
    x = x.astype(np.float32)
    ss = np.float32(1.0) / np.sqrt(np.float32((x*x).sum()) / np.float32(len(x)) + np.float32(1e-5))
    return (g * (ss * x)).astype(np.float32)

def softmax(x):
    e = np.exp(x - x.max()); return (e / e.sum()).astype(np.float32)

def ref_forward(W, cfg, tokens):
    dim, hid, L, H, KVH, V, S, gs = (cfg[k] for k in
        ("dim","hidden","layers","heads","kv_heads","vocab","seq","gs"))
    hs = dim // H; kv_dim = dim * KVH // H; kv_mul = H // KVH
    kc = np.zeros((L, S, kv_dim), np.float32)
    vc = np.zeros((L, S, kv_dim), np.float32)
    out = []
    for pos, tok in enumerate(tokens):
        # dequant embedding row
        q = W["tok_emb_q"][tok].reshape(dim//gs, gs).astype(np.float32)
        s = W["tok_emb_s"][tok]
        x = (q * s[:, None]).reshape(dim).astype(np.float32)
        for l in range(L):
            xb = rmsnorm(x, W["rms_att"][l])
            qv = qmatmul(W["wq_q"][l], W["wq_s"][l], xb, gs)
            k  = qmatmul(W["wk_q"][l], W["wk_s"][l], xb, gs)
            v  = qmatmul(W["wv_q"][l], W["wv_s"][l], xb, gs)
            for i in range(0, dim, 2):
                hd = i % hs
                freq = np.float32(10000.0) ** np.float32(-hd / hs)
                val = np.float32(pos) * freq
                fcr, fci = np.cos(val, dtype=np.float32), np.sin(val, dtype=np.float32)
                for vec in ([qv, k] if i < kv_dim else [qv]):
                    v0, v1 = vec[i], vec[i+1]
                    vec[i]   = v0*fcr - v1*fci
                    vec[i+1] = v0*fci + v1*fcr
            kc[l, pos] = k; vc[l, pos] = v
            xb2 = np.zeros(dim, np.float32)
            for h in range(H):
                qh = qv[h*hs:(h+1)*hs]
                ks = kc[l, :pos+1, (h//kv_mul)*hs:(h//kv_mul+1)*hs]
                att = softmax((ks @ qh) / np.sqrt(np.float32(hs)))
                xb2[h*hs:(h+1)*hs] = att @ vc[l, :pos+1, (h//kv_mul)*hs:(h//kv_mul+1)*hs]
            x = x + qmatmul(W["wo_q"][l], W["wo_s"][l], xb2, gs)
            xb = rmsnorm(x, W["rms_ffn"][l])
            hb  = qmatmul(W["w1_q"][l], W["w1_s"][l], xb, gs)
            hb2 = qmatmul(W["w3_q"][l], W["w3_s"][l], xb, gs)
            hb = (hb * (1.0/(1.0+np.exp(-hb))) * hb2).astype(np.float32)
            x = x + qmatmul(W["w2_q"][l], W["w2_s"][l], hb, gs)
        x = rmsnorm(x, W["rms_final"])
        out.append(qmatmul(W["wcls_q"], W["wcls_s"], x, gs))
    return np.array(out, np.float32)

def build(bits, gs, path_model, path_ref):
    rng = np.random.default_rng(42)
    cfg = dict(dim=64, hidden=192, layers=2, heads=2, kv_heads=2,
               vocab=512, seq=32, gs=gs)
    dim, hid, L, H, KVH, V, S = (cfg[k] for k in
        ("dim","hidden","layers","heads","kv_heads","vocab","seq"))
    kv_dim = dim * KVH // H
    def rand(*shape): return (rng.standard_normal(shape) * 0.08).astype(np.float32)

    W = {}
    W["rms_att"] = 1.0 + rand(L, dim) * 0; W["rms_att"] += rand(L, dim)
    W["rms_ffn"] = 1.0 + rand(L, dim)
    W["rms_final"] = (1.0 + rand(dim)).astype(np.float32)
    parts = []
    def addq(name, w2d, per_layer_rows=None):
        s, packed, q = quant_weights(w2d, bits, gs)
        if per_layer_rows:
            W[name+"_q"] = q.reshape(L, per_layer_rows, w2d.shape[1])
            W[name+"_s"] = s.reshape(L, per_layer_rows, w2d.shape[1]//gs)
        else:
            W[name+"_q"], W[name+"_s"] = q, s
        parts.append(s.astype(np.float32).tobytes()); parts.append(packed)

    header = b""  # built at the end
    # tok_offset/tok_size patched later
    blob = bytearray(64)
    norms = (W["rms_att"].astype(np.float32).tobytes() +
             W["rms_ffn"].astype(np.float32).tobytes() +
             W["rms_final"].tobytes())

    addq("tok_emb", rand(V, dim))
    addq("wq", rand(L*dim, dim), dim)
    addq("wk", rand(L*kv_dim, dim), kv_dim)
    addq("wv", rand(L*kv_dim, dim), kv_dim)
    addq("wo", rand(L*dim, dim), dim)
    addq("w1", rand(L*hid, dim), hid)
    addq("w2", rand(L*dim, hid), dim)
    addq("w3", rand(L*hid, dim), hid)
    W["wcls_q"], W["wcls_s"] = W["tok_emb_q"], W["tok_emb_s"]

    # minimal tokenizer blob: max_len + vocab entries "t<i>"
    tok = bytearray(struct.pack("<i", 8))
    for i in range(V):
        piece = f"t{i}".encode()
        tok += struct.pack("<fi", 0.0, len(piece)) + piece + b"\0"

    weights = norms + b"".join(parts)
    tok_offset = 64 + len(weights)
    hdr = struct.pack("<IIiiiiiiiiiiiiii", MAGIC, 2, bits, gs, dim, hid, L, H,
                      KVH, V, S, 1, tok_offset, len(tok), L, 3)
    blob[:len(hdr)] = hdr
    with open(path_model, "wb") as f:
        f.write(bytes(blob) + weights + tok)

    tokens = [5, 17, 42, 9, 100, 3]
    logits = ref_forward(W, cfg, tokens)
    with open(path_ref, "wb") as f:
        f.write(struct.pack("<ii", len(tokens), V))
        f.write(np.array(tokens, np.int32).tobytes())
        f.write(logits.tobytes())
    print(f"bits={bits}: wrote {path_model} ({64+len(weights)+len(tok)} bytes), "
          f"ref logits range [{logits.min():.3f}, {logits.max():.3f}]")

if __name__ == "__main__":
    build(8, 32, "synth_q8.bin", "ref_q8.bin")
    build(4, 32, "synth_q4.bin", "ref_q4.bin")
