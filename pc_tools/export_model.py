#!/usr/bin/env python3
"""
export_model.py — convert a llama2.c fp32 checkpoint (.bin v0 format,
e.g. stories15M.bin) + tokenizer.bin into a quantized ESPL flash image
for the ESP32-S3 firmware. NumPy only, no torch.

Get the inputs (on your PC):
  wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
  wget https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin

Usage:
  python3 export_model.py stories15M.bin tokenizer.bin model_esp.bin --bits 4
  python3 export_model.py stories15M.bin tokenizer.bin model_esp.bin --bits 8

Sizes for stories15M:  --bits 4 -> ~9 MB   --bits 8 -> ~16.5 MB
A 16MB-flash board (e.g. Waveshare ESP32-S3-Touch-LCD-5) needs --bits 4.
"""
import numpy as np, struct, sys, argparse

MAGIC = 0x4C505345

def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def quant(w, bits, gs):
    rows, n = w.shape
    assert n % gs == 0, f"cols {n} not divisible by group size {gs}"
    g = w.reshape(rows, n // gs, gs)
    wmax = np.abs(g).max(axis=2)
    qmax = 127.0 if bits == 8 else 7.0
    scale = np.where(wmax == 0, 1.0, wmax / qmax).astype(np.float32)
    q = np.clip(round_away(g / scale[..., None]), -qmax, qmax).astype(np.int32)
    err = np.abs(q * scale[..., None] - g).max()
    if bits == 8:
        packed = q.astype(np.int8).reshape(rows, n).tobytes()
    else:
        qn = (q + 8).astype(np.uint8).reshape(rows, n)
        packed = (qn[:, 0::2] | (qn[:, 1::2] << 4)).tobytes()
    return scale.tobytes() + packed, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("tokenizer")
    ap.add_argument("out"); ap.add_argument("--bits", type=int, default=4,
                    choices=[4, 8])
    ap.add_argument("--gs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=0,
                    help="override max seq len (smaller = less KV RAM)")
    a = ap.parse_args()

    raw = open(a.checkpoint, "rb").read()
    dim, hid, L, H, KVH, V, S = struct.unpack("<7i", raw[:28])
    shared = 1 if V > 0 else 0
    V = abs(V)
    if a.seq: S = min(S, a.seq)
    print(f"config: dim={dim} hidden={hid} layers={L} heads={H} "
          f"kv_heads={KVH} vocab={V} seq={S} shared_cls={shared}")
    if dim % a.gs or hid % a.gs:
        sys.exit(f"group size {a.gs} must divide dim ({dim}) and hidden ({hid})")

    f32 = np.frombuffer(raw, np.float32, offset=28)
    pos = 0
    def take(*shape):
        nonlocal pos
        cnt = int(np.prod(shape))
        t = f32[pos:pos+cnt].reshape(shape).astype(np.float32)
        pos += cnt
        return t

    kv_dim = dim * KVH // H
    hs = dim // H
    tok_emb = take(V, dim)
    rms_att = take(L, dim)
    wq = take(L, dim, dim);    wk = take(L, kv_dim, dim)
    wv = take(L, kv_dim, dim); wo = take(L, dim, dim)
    rms_ffn = take(L, dim)
    w1 = take(L, hid, dim); w2 = take(L, dim, hid); w3 = take(L, hid, dim)
    rms_final = take(dim)
    # skip legacy freq_cis tables (computed on device)
    orig_seq = struct.unpack("<i", raw[24:28])[0]
    pos += orig_seq * hs  # real + imag, each seq*(hs/2)
    wcls = tok_emb if shared else take(V, dim)

    norms = rms_att.tobytes() + rms_ffn.tobytes() + rms_final.tobytes()
    parts, errs = [], []
    for name, w in [("tok_emb", tok_emb),
                    ("wq", wq.reshape(L*dim, dim)),
                    ("wk", wk.reshape(L*kv_dim, dim)),
                    ("wv", wv.reshape(L*kv_dim, dim)),
                    ("wo", wo.reshape(L*dim, dim)),
                    ("w1", w1.reshape(L*hid, dim)),
                    ("w2", w2.reshape(L*dim, hid)),
                    ("w3", w3.reshape(L*hid, dim))]:
        blob, err = quant(w, a.bits, a.gs)
        parts.append(blob); errs.append((name, err))
    if not shared:
        blob, err = quant(wcls, a.bits, a.gs)
        parts.append(blob); errs.append(("wcls", err))
    for n, e in errs: print(f"  {n}: max quant err {e:.5f}")

    tok = open(a.tokenizer, "rb").read()
    # repack tokenizer with NUL-terminated strings for zero-copy reads
    max_len, = struct.unpack("<i", tok[:4])
    p = 4; out_tok = bytearray(struct.pack("<i", max_len))
    for _ in range(V):
        score, ln = struct.unpack("<fi", tok[p:p+8]); p += 8
        piece = tok[p:p+ln]; p += ln
        out_tok += struct.pack("<fi", score, ln) + piece + b"\0"

    weights = norms + b"".join(parts)
    tok_offset = 64 + len(weights)
    hdr = struct.pack("<IIiiiiiiiiiiiiii", MAGIC, 2, a.bits, a.gs, dim, hid,
                      L, H, KVH, V, S, shared, tok_offset, len(out_tok), L, 3)
    image = hdr + b"\0" * (64 - len(hdr)) + weights + bytes(out_tok)
    open(a.out, "wb").write(image)
    print(f"\nwrote {a.out}: {len(image)/1e6:.2f} MB "
          f"(weights {len(weights)/1e6:.2f} MB, tokenizer {len(out_tok)/1e3:.0f} KB)")
    print(f"flash with: esptool.py write_flash 0x1F0000 {a.out}")

if __name__ == "__main__":
    main()
