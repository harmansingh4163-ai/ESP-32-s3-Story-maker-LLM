#!/usr/bin/env python3
"""
split_image.py — split a full ESPL model image into two images for the
two-board pipeline:

  worker image: layers [0, K)            -> board A (the coprocessor)
  head image:   embedding + layers [K, L) + classifier + tokenizer
                                          -> board B (the user-facing board)

  python3 split_image.py model_esp.bin 3 worker.bin head.bin

Per token, board B embeds, ships the activation vector to A over UART,
A runs its layers, ships it back, B finishes and samples.
"""
import struct, sys
import numpy as np

MAGIC = 0x4C505345

def parse(path):
    raw = open(path, "rb").read()
    f = struct.unpack("<IIiiiiiiiiiiiiii", raw[:64])
    assert f[0] == MAGIC and f[1] == 2, "not a v2 ESPL image"
    keys = ("magic","ver","bits","gs","dim","hid","L","H","KVH","V","S",
            "shared","tok_off","tok_size","local","flags")
    h = dict(zip(keys, f))
    assert h["local"] == h["L"] and h["flags"] == 3, "input must be a full image"
    return h, raw

def qsize(rows, n, bits, gs):
    return rows * (n // gs) * 4 + (rows * n if bits == 8 else rows * n // 2)

def main():
    src, K, out_w, out_h = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    h, raw = parse(src)
    bits, gs = h["bits"], h["gs"]
    dim, hid, L, V = h["dim"], h["hid"], h["L"], h["V"]
    kv_dim = dim * h["KVH"] // h["H"]
    assert 0 < K < L, f"split point must be in 1..{L-1}"

    p = 64
    def take(nbytes):
        nonlocal p
        b = raw[p:p+nbytes]; p += nbytes
        return b
    rms_att = take(L * dim * 4)
    rms_ffn = take(L * dim * 4)
    rms_final = take(dim * 4)
    tok_emb = take(qsize(V, dim, bits, gs))
    tensors = {}
    for name, rows, n in [("wq", dim, dim), ("wk", kv_dim, dim),
                          ("wv", kv_dim, dim), ("wo", dim, dim),
                          ("w1", hid, dim), ("w2", dim, hid), ("w3", hid, dim)]:
        srows = L * rows
        blob = take(qsize(srows, n, bits, gs))
        ssz = srows * (n // gs) * 4
        scales, packed = blob[:ssz], blob[ssz:]
        per_s = rows * (n // gs) * 4
        per_q = rows * n if bits == 8 else rows * n // 2
        tensors[name] = [(scales[l*per_s:(l+1)*per_s], packed[l*per_q:(l+1)*per_q])
                         for l in range(L)]
    if h["shared"] == 0:
        take(qsize(V, dim, bits, gs))   # skip stored wcls
    assert p == h["tok_off"], f"layout walk mismatch: {p} vs {h['tok_off']}"
    tok = raw[h["tok_off"]:h["tok_off"]+h["tok_size"]]

    def slice_norm(blob, l0, l1):
        return blob[l0*dim*4:l1*dim*4]
    def build(l0, l1, flags, with_tok):
        # layout matches qmat_attach: per tensor, scales of the layer range
        # concatenated, then packed weights of the range
        parts = [slice_norm(rms_att, l0, l1), slice_norm(rms_ffn, l0, l1)]
        if flags & 2: parts.append(rms_final)
        if flags & 1: parts.append(tok_emb)
        for name in ["wq","wk","wv","wo","w1","w2","w3"]:
            for l in range(l0, l1): parts.append(tensors[name][l][0])
            for l in range(l0, l1): parts.append(tensors[name][l][1])
        weights = b"".join(parts)
        tk = tok if with_tok else b""
        tok_off = 64 + len(weights) if with_tok else 0
        hdr = struct.pack("<IIiiiiiiiiiiiiii", MAGIC, 2, bits, gs, dim, hid,
                          L, h["H"], h["KVH"], V, h["S"], h["shared"],
                          tok_off, len(tk), l1 - l0, flags)
        return hdr + b"\0"*(64-len(hdr)) + weights + tk

    w = build(0, K, 0, False)            # worker: layers only
    hd = build(K, L, 3, True)            # head: emb + layers + head + tokenizer
    open(out_w, "wb").write(w)
    open(out_h, "wb").write(hd)
    print(f"worker {out_w}: layers 0..{K-1}, {len(w)/1e6:.2f} MB")
    print(f"head   {out_h}: emb + layers {K}..{L-1} + cls, {len(hd)/1e6:.2f} MB")

if __name__ == "__main__":
    main()
