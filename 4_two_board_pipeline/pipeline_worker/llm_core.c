/*
 * llm_core.c — quantized Llama-2 inference, portable C99.
 * Architecture follows llama2.c (Karpathy, MIT) with a custom Q4/Q8
 * weight format designed for ESP32 flash memory-mapping.
 *
 * Tactics implemented here:
 *   #1 weights are read through a const pointer -> works directly on
 *      memory-mapped flash, zero RAM for weights
 *   #2 INT8 and packed INT4 weights, INT8 activations, integer dot products
 *   #3 the hot loop lives in llm_matmul_rows(); drop PIE SIMD asm there
 *   #4 llm_parallel_matmul hook lets firmware split rows across cores
 */
#include "llm_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void* (*llm_alloc_big)(size_t)   = NULL;
void* (*llm_alloc_small)(size_t) = NULL;
void (*llm_parallel_matmul)(const MMJob*) = NULL;

static void* abig(size_t n)   { return llm_alloc_big   ? llm_alloc_big(n)   : malloc(n); }
static void* asmall(size_t n) { return llm_alloc_small ? llm_alloc_small(n) : malloc(n); }

/* ================= quantization ================= */

/* quantize fp32 activations to int8, one scale per group */
static void quantize_act(const float* x, int n, int gs, int8_t* q, float* s) {
  for (int g = 0; g < n / gs; g++) {
    float wmax = 0.f;
    const float* xg = x + g * gs;
    for (int i = 0; i < gs; i++) { float a = fabsf(xg[i]); if (a > wmax) wmax = a; }
    float scale = (wmax == 0.f) ? 1.f : wmax / 127.f;
    s[g] = scale;
    int8_t* qg = q + g * gs;
    for (int i = 0; i < gs; i++) qg[i] = (int8_t)roundf(xg[i] / scale);
  }
}

/* ================= matmul (the hot path) =================
 * out[r] = sum_j W[r][j] * x[j],  W quantized (int8 or packed int4),
 * x quantized to int8. Integer MACs accumulate exactly in int32 per
 * group, then scale to fp32.
 *
 * TACTIC #3 PLUG-IN POINT: replace the inner loops below with ESP32-S3
 * PIE 128-bit SIMD (ee.vmulas.s8.qacc et al) for a further 3-5x.
 */
void llm_matmul_rows(float* out, const QMat* w, int bits, int gs,
                     const int8_t* qx, const float* sx, int n, int r0, int r1) {
  const int ngroups = n / gs;
  if (bits == 8) {
    for (int r = r0; r < r1; r++) {
      const int8_t* wr = (const int8_t*)w->q + (size_t)r * n;
      const float*  sr = w->s + (size_t)r * ngroups;
      float acc = 0.f;
      for (int g = 0; g < ngroups; g++) {
        const int8_t* wg = wr + g * gs;
        const int8_t* xg = qx + g * gs;
        int32_t iv = 0;
        for (int i = 0; i < gs; i += 4) {       /* unrolled x4 */
          iv += (int32_t)wg[i]   * xg[i]
              + (int32_t)wg[i+1] * xg[i+1]
              + (int32_t)wg[i+2] * xg[i+2]
              + (int32_t)wg[i+3] * xg[i+3];
        }
        acc += (float)iv * sr[g] * sx[g];
      }
      out[r] = acc;
    }
  } else { /* int4: two weights per byte, stored as (q+8) nibbles */
    for (int r = r0; r < r1; r++) {
      const uint8_t* wr = w->q + (size_t)r * (n / 2);
      const float*   sr = w->s + (size_t)r * ngroups;
      float acc = 0.f;
      for (int g = 0; g < ngroups; g++) {
        const uint8_t* wg = wr + g * (gs / 2);
        const int8_t*  xg = qx + g * gs;
        int32_t iv = 0;
        for (int i = 0; i < gs / 2; i++) {
          uint8_t b = wg[i];
          iv += ((int32_t)(b & 0x0F) - 8) * xg[2*i]
              + ((int32_t)(b >> 4)   - 8) * xg[2*i + 1];
        }
        acc += (float)iv * sr[g] * sx[g];
      }
      out[r] = acc;
    }
  }
}

static void matmul(LLM* m, float* out, const QMat* w,
                   const int8_t* qx, const float* sx, int n, int d) {
  if (llm_parallel_matmul) {
    MMJob j = { out, w, m->h.bits, m->h.gs, qx, sx, n, d };
    llm_parallel_matmul(&j);
  } else {
    llm_matmul_rows(out, w, m->h.bits, m->h.gs, qx, sx, n, 0, d);
  }
}

/* dequantize one row of a QMat (used for the token embedding) */
static void dequant_row(const LLM* m, const QMat* w, int row, int n, float* out) {
  const int gs = m->h.gs, ngroups = n / gs;
  const float* sr = w->s + (size_t)row * ngroups;
  if (m->h.bits == 8) {
    const int8_t* qr = (const int8_t*)w->q + (size_t)row * n;
    for (int g = 0; g < ngroups; g++)
      for (int i = 0; i < gs; i++)
        out[g*gs + i] = (float)qr[g*gs + i] * sr[g];
  } else {
    const uint8_t* qr = w->q + (size_t)row * (n / 2);
    for (int g = 0; g < ngroups; g++)
      for (int i = 0; i < gs/2; i++) {
        uint8_t b = qr[g*(gs/2) + i];
        out[g*gs + 2*i]   = (float)((int)(b & 0x0F) - 8) * sr[g];
        out[g*gs + 2*i+1] = (float)((int)(b >> 4)   - 8) * sr[g];
      }
  }
}

/* ================= layers ================= */

static void rmsnorm(float* o, const float* x, const float* g, int n) {
  float ss = 0.f;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  ss = 1.f / sqrtf(ss / n + 1e-5f);
  for (int i = 0; i < n; i++) o[i] = g[i] * (ss * x[i]);
}

static void softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
  float sum = 0.f;
  for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

void llm_embed(LLM* m, int token) {
  dequant_row(m, &m->tok_emb, token, m->h.dim, m->x);
}

void llm_layers(LLM* m, int pos) {
  const LLMHeader* p = &m->h;
  const int dim = p->dim, hidden = p->hidden_dim, gs = p->gs;
  const int kv_dim = m->kv_dim, hs = m->head_size;
  const int kv_mul = p->n_heads / p->n_kv_heads;
  float* x = m->x;

  for (int l = 0; l < p->local_layers; l++) {
    rmsnorm(m->xb, x, m->rms_att + (size_t)l * dim, dim);

    quantize_act(m->xb, dim, gs, m->qbuf, m->sbuf);
    float* k = m->key_cache + ((size_t)l * p->seq_len + pos) * kv_dim;
    float* v = m->val_cache + ((size_t)l * p->seq_len + pos) * kv_dim;
    /* offset QMat views into the per-layer weight blocks */
    QMat wq = { m->wq.s + (size_t)l*dim*(dim/gs),
                m->wq.q + (size_t)l*dim*dim*p->bits/8 };
    QMat wk = { m->wk.s + (size_t)l*kv_dim*(dim/gs),
                m->wk.q + (size_t)l*kv_dim*dim*p->bits/8 };
    QMat wv = { m->wv.s + (size_t)l*kv_dim*(dim/gs),
                m->wv.q + (size_t)l*kv_dim*dim*p->bits/8 };
    matmul(m, m->q, &wq, m->qbuf, m->sbuf, dim, dim);
    matmul(m, k,    &wk, m->qbuf, m->sbuf, dim, kv_dim);
    matmul(m, v,    &wv, m->qbuf, m->sbuf, dim, kv_dim);

    /* RoPE: rotate q (full dim) and k (kv_dim) */
    for (int i = 0; i < dim; i += 2) {
      int hd = i % hs;
      float freq = powf(10000.0f, -(float)hd / (float)hs);
      float val = pos * freq, fcr = cosf(val), fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int vv = 0; vv < rotn; vv++) {
        float* vec = vv == 0 ? m->q : k;
        float v0 = vec[i], v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
      }
    }

    /* attention */
    for (int h = 0; h < p->n_heads; h++) {
      const float* qh = m->q + h * hs;
      float* att = m->att;
      for (int t = 0; t <= pos; t++) {
        const float* kt = m->key_cache + ((size_t)l * p->seq_len + t) * kv_dim
                          + (h / kv_mul) * hs;
        float sc = 0.f;
        for (int i = 0; i < hs; i++) sc += qh[i] * kt[i];
        att[t] = sc / sqrtf((float)hs);
      }
      softmax(att, pos + 1);
      float* xbh = m->xb + h * hs;
      memset(xbh, 0, hs * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        const float* vt = m->val_cache + ((size_t)l * p->seq_len + t) * kv_dim
                          + (h / kv_mul) * hs;
        float a = att[t];
        for (int i = 0; i < hs; i++) xbh[i] += a * vt[i];
      }
    }

    quantize_act(m->xb, dim, gs, m->qbuf, m->sbuf);
    QMat wo = { m->wo.s + (size_t)l*dim*(dim/gs),
                m->wo.q + (size_t)l*dim*dim*p->bits/8 };
    matmul(m, m->xb2, &wo, m->qbuf, m->sbuf, dim, dim);
    for (int i = 0; i < dim; i++) x[i] += m->xb2[i];

    /* FFN: w2( silu(w1 x) * w3 x ) */
    rmsnorm(m->xb, x, m->rms_ffn + (size_t)l * dim, dim);
    quantize_act(m->xb, dim, gs, m->qbuf, m->sbuf);
    QMat w1 = { m->w1.s + (size_t)l*hidden*(dim/gs),
                m->w1.q + (size_t)l*hidden*dim*p->bits/8 };
    QMat w3 = { m->w3.s + (size_t)l*hidden*(dim/gs),
                m->w3.q + (size_t)l*hidden*dim*p->bits/8 };
    matmul(m, m->hb,  &w1, m->qbuf, m->sbuf, dim, hidden);
    matmul(m, m->hb2, &w3, m->qbuf, m->sbuf, dim, hidden);
    for (int i = 0; i < hidden; i++) {
      float val = m->hb[i];
      val *= 1.0f / (1.0f + expf(-val));   /* SiLU */
      m->hb[i] = val * m->hb2[i];
    }
    quantize_act(m->hb, hidden, gs, m->qbuf, m->sbuf);
    QMat w2 = { m->w2.s + (size_t)l*dim*(hidden/gs),
                m->w2.q + (size_t)l*dim*hidden*p->bits/8 };
    matmul(m, m->xb, &w2, m->qbuf, m->sbuf, hidden, dim);
    for (int i = 0; i < dim; i++) x[i] += m->xb[i];
  }
}

void llm_head(LLM* m) {
  const LLMHeader* p = &m->h;
  rmsnorm(m->x, m->x, m->rms_final, p->dim);
  quantize_act(m->x, p->dim, p->gs, m->qbuf, m->sbuf);
  matmul(m, m->logits, &m->wcls, m->qbuf, m->sbuf, p->dim, p->vocab_size);
}

void llm_forward(LLM* m, int token, int pos) {
  llm_embed(m, token);
  llm_layers(m, pos);
  llm_head(m);
}

/* ================= loader ================= */

static const uint8_t* qmat_attach(QMat* t, const uint8_t* ptr,
                                  size_t rows, size_t n, int bits, int gs) {
  t->s = (const float*)ptr;
  ptr += rows * (n / gs) * sizeof(float);
  t->q = ptr;
  ptr += (bits == 8) ? rows * n : rows * n / 2;
  return ptr;
}

int llm_init(LLM* m, const uint8_t* image, size_t image_size) {
  memcpy(&m->h, image, sizeof(LLMHeader));
  if (m->h.magic != LLM_MAGIC || m->h.version != 2) return -1;
  if (m->h.bits != 4 && m->h.bits != 8) return -2;
  const LLMHeader* p = &m->h;
  if (p->dim % p->gs || p->hidden_dim % p->gs) return -3;
  m->base = image;
  m->kv_dim = p->dim * p->n_kv_heads / p->n_heads;
  m->head_size = p->dim / p->n_heads;

  const int LL = p->local_layers;
  const uint8_t* ptr = image + 64; /* header padded to 64 bytes */
  m->rms_att   = (const float*)ptr; ptr += (size_t)LL * p->dim * 4;
  m->rms_ffn   = (const float*)ptr; ptr += (size_t)LL * p->dim * 4;
  if (p->flags & LLM_HAS_HEAD) {
    m->rms_final = (const float*)ptr; ptr += (size_t)p->dim * 4;
  } else m->rms_final = NULL;

  int b = p->bits, gs = p->gs;
  if (p->flags & LLM_HAS_EMB)
    ptr = qmat_attach(&m->tok_emb, ptr, p->vocab_size, p->dim, b, gs);
  ptr = qmat_attach(&m->wq, ptr, (size_t)LL * p->dim,      p->dim, b, gs);
  ptr = qmat_attach(&m->wk, ptr, (size_t)LL * m->kv_dim,   p->dim, b, gs);
  ptr = qmat_attach(&m->wv, ptr, (size_t)LL * m->kv_dim,   p->dim, b, gs);
  ptr = qmat_attach(&m->wo, ptr, (size_t)LL * p->dim,      p->dim, b, gs);
  ptr = qmat_attach(&m->w1, ptr, (size_t)LL * p->hidden_dim, p->dim, b, gs);
  ptr = qmat_attach(&m->w2, ptr, (size_t)LL * p->dim,      p->hidden_dim, b, gs);
  ptr = qmat_attach(&m->w3, ptr, (size_t)LL * p->hidden_dim, p->dim, b, gs);
  if (p->flags & LLM_HAS_HEAD) {
    if (p->shared_cls) {
      if (!(p->flags & LLM_HAS_EMB)) return -5; /* shared cls needs emb here */
      m->wcls = m->tok_emb;
    } else ptr = qmat_attach(&m->wcls, ptr, p->vocab_size, p->dim, b, gs);
  }

  /* runtime buffers */
  int dim = p->dim, hidden = p->hidden_dim;
  int maxn = dim > hidden ? dim : hidden;
  m->x    = (float*)asmall(dim * 4);  m->xb  = (float*)asmall(dim * 4);
  m->xb2  = (float*)asmall(dim * 4);  m->hb  = (float*)asmall(hidden * 4);
  m->hb2  = (float*)asmall(hidden * 4); m->q = (float*)asmall(dim * 4);
  m->att  = (float*)asmall(p->seq_len * 4);
  m->qbuf = (int8_t*)asmall(maxn);    m->sbuf = (float*)asmall((maxn / gs) * 4);
  m->logits = (p->flags & LLM_HAS_HEAD)
                ? (float*)abig((size_t)p->vocab_size * 4) : (float*)m->xb;
  m->key_cache = (float*)abig((size_t)LL * p->seq_len * m->kv_dim * 4);
  m->val_cache = (float*)abig((size_t)LL * p->seq_len * m->kv_dim * 4);
  if (!m->x || !m->xb || !m->xb2 || !m->hb || !m->hb2 || !m->q || !m->att ||
      !m->qbuf || !m->sbuf || !m->logits || !m->key_cache || !m->val_cache)
    return -4;
  (void)image_size;
  return 0;
}

/* ================= tokenizer (llama2.c BPE format) ================= */

typedef struct { const char* str; int id; } TokIndex;
static TokIndex* g_sorted = NULL;
static const char** g_pieces = NULL;
static float* g_scores = NULL;
static int g_vocab = 0;
static unsigned g_max_token_len = 0;
static char g_piece_buf[8];

static int cmp_tok(const void* a, const void* b) {
  return strcmp(((const TokIndex*)a)->str, ((const TokIndex*)b)->str);
}

int llm_tok_init(LLM* m) {
  const uint8_t* p = m->base + m->h.tok_offset;
  g_vocab = m->h.vocab_size;
  memcpy(&g_max_token_len, p, 4); p += 4;
  g_pieces = (const char**)abig(g_vocab * sizeof(char*));
  g_scores = (float*)abig(g_vocab * sizeof(float));
  g_sorted = (TokIndex*)abig(g_vocab * sizeof(TokIndex));
  if (!g_pieces || !g_scores || !g_sorted) return -1;
  for (int i = 0; i < g_vocab; i++) {
    memcpy(&g_scores[i], p, 4); p += 4;
    int32_t len; memcpy(&len, p, 4); p += 4;
    g_pieces[i] = (const char*)p;       /* NUL-terminated in image */
    p += len + 1;
    g_sorted[i].str = g_pieces[i];
    g_sorted[i].id = i;
  }
  qsort(g_sorted, g_vocab, sizeof(TokIndex), cmp_tok);
  return 0;
}

static int str_lookup(const char* s) {
  int lo = 0, hi = g_vocab - 1;
  while (lo <= hi) {
    int mid = (lo + hi) / 2;
    int c = strcmp(s, g_sorted[mid].str);
    if (c == 0) return g_sorted[mid].id;
    if (c < 0) hi = mid - 1; else lo = mid + 1;
  }
  return -1;
}

int llm_encode(LLM* m, const char* text, int bos, int eos, int* tokens, int max) {
  (void)m;
  int n = 0;
  char* buf = (char*)malloc(g_max_token_len * 2 + 3);
  if (bos && n < max) tokens[n++] = 1;
  if (text[0] != '\0') {
    int sp = str_lookup(" ");
    if (sp >= 0 && n < max) tokens[n++] = sp;  /* dummy prefix space */
  }
  /* UTF-8 aware: accumulate bytes of a codepoint, then lookup */
  size_t blen = 0;
  for (const char* c = text; *c; c++) {
    if ((*c & 0xC0) != 0x80) blen = 0;           /* not a continuation byte */
    buf[blen++] = *c; buf[blen] = '\0';
    if ((*(c+1) & 0xC0) == 0x80 && blen < 4) continue;
    int id = str_lookup(buf);
    if (id != -1) { if (n < max) tokens[n++] = id; }
    else for (size_t i = 0; i < blen; i++)        /* byte fallback */
      if (n < max) tokens[n++] = (unsigned char)buf[i] + 3;
    blen = 0;
  }
  /* merge loop */
  while (1) {
    float best_score = -1e10f; int best_id = -1, best_idx = -1;
    for (int i = 0; i < n - 1; i++) {
      snprintf(buf, g_max_token_len * 2 + 3, "%s%s",
               g_pieces[tokens[i]], g_pieces[tokens[i+1]]);
      int id = str_lookup(buf);
      if (id != -1 && g_scores[id] > best_score) {
        best_score = g_scores[id]; best_id = id; best_idx = i;
      }
    }
    if (best_idx == -1) break;
    tokens[best_idx] = best_id;
    memmove(&tokens[best_idx+1], &tokens[best_idx+2],
            (n - best_idx - 2) * sizeof(int));
    n--;
  }
  if (eos && n < max) tokens[n++] = 2;
  free(buf);
  return n;
}

const char* llm_decode(LLM* m, int prev_token, int token) {
  (void)m;
  const char* piece = g_pieces[token];
  if (prev_token == 1 && piece[0] == ' ') piece++; /* strip space after BOS */
  unsigned char byte;
  if (sscanf(piece, "<0x%02hhX>", &byte) == 1) {
    g_piece_buf[0] = (char)byte; g_piece_buf[1] = '\0';
    return g_piece_buf;
  }
  return piece;
}

/* ================= sampler ================= */

static unsigned int rnd_u32(uint64_t* s) {
  *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
  return (*s * 0x2545F4914F6CDD1DULL) >> 32;
}
static float rnd_f32(uint64_t* s) { return (rnd_u32(s) >> 8) / 16777216.0f; }

int llm_sample(LLM* m, Sampler* sp) {
  int v = m->h.vocab_size;
  float* lg = m->logits;
  if (sp->temperature <= 0.f) {                  /* greedy */
    int best = 0;
    for (int i = 1; i < v; i++) if (lg[i] > lg[best]) best = i;
    return best;
  }
  for (int i = 0; i < v; i++) lg[i] /= sp->temperature;
  softmax(lg, v);
  float r = rnd_f32(&sp->rng);
  if (sp->topp <= 0.f || sp->topp >= 1.f) {       /* plain multinomial */
    float cdf = 0.f;
    for (int i = 0; i < v; i++) { cdf += lg[i]; if (r < cdf) return i; }
    return v - 1;
  }
  /* top-p: collect candidates above a floor, partial-sort by prob */
  /* simple O(v log v)-ish approach with an index array in logits tail is
     too memory hungry; do selection over a cutoff like llama2.c */
  float cutoff = (1.0f - sp->topp) / (v - 1);
  /* count + crude insertion into a small fixed pool */
  #define POOL 512
  static int   idx[POOL];
  static float prb[POOL];
  int np = 0;
  for (int i = 0; i < v; i++) {
    if (lg[i] < cutoff) continue;
    /* insert sorted desc, keep top POOL */
    int j = np < POOL ? np : POOL - 1;
    if (np < POOL) np++;
    while (j > 0 && prb[j-1] < lg[i]) {
      prb[j] = prb[j-1]; idx[j] = idx[j-1]; j--;
    }
    prb[j] = lg[i]; idx[j] = i;
  }
  float cum = 0.f; int last = np - 1;
  for (int i = 0; i < np; i++) {
    cum += prb[i];
    if (cum > sp->topp) { last = i; break; }
  }
  float rr = r * cum, cdf = 0.f;
  for (int i = 0; i <= last; i++) {
    cdf += prb[i];
    if (rr < cdf) return idx[i];
  }
  return idx[last];
}
