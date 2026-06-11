/*
 * llm_core.h — portable quantized Llama-2-architecture inference core.
 * Compiles on host (testing) and ESP32-S3 (deployment). C99, no deps.
 */
#ifndef LLM_CORE_H
#define LLM_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LLM_MAGIC 0x4C505345u /* "ESPL" */

typedef struct {
  uint32_t magic, version;      /* version 2 */
  int32_t bits;        /* 4 or 8 */
  int32_t gs;          /* quantization group size */
  int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
  int32_t shared_cls;  /* 1 = classifier shares token embedding */
  int32_t tok_offset, tok_size; /* tokenizer blob location (0 if absent) */
  int32_t local_layers;         /* layers stored in THIS image */
  int32_t flags;                /* bit0: has embedding, bit1: has head */
} LLMHeader;

#define LLM_HAS_EMB  1
#define LLM_HAS_HEAD 2

typedef struct {           /* one quantized matrix, data lives in flash/mmap */
  const float* s;          /* group scales, rows * (n/gs) */
  const uint8_t* q;        /* packed weights: int8, or 2x int4 per byte    */
} QMat;

typedef struct {
  LLMHeader h;
  const uint8_t* base;     /* mmap'd image base */
  /* fp32 norm weights */
  const float *rms_att, *rms_ffn, *rms_final;
  /* quantized tensors */
  QMat tok_emb, wq, wk, wv, wo, w1, w2, w3, wcls;
  /* runtime state (RAM) */
  float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits;
  float *key_cache, *val_cache;       /* big: prefer PSRAM */
  int8_t* qbuf; float* sbuf;          /* quantized activation scratch */
  int kv_dim, head_size;
} LLM;

/* ---- memory hooks: set before llm_init (defaults to malloc) ---- */
extern void* (*llm_alloc_big)(size_t);    /* KV cache, logits -> PSRAM   */
extern void* (*llm_alloc_small)(size_t);  /* activations -> internal RAM */

/* ---- parallel matmul hook (tactic 4): if set, called instead of the
   serial loop. Implementation must compute all rows [0, d) of the matmul,
   typically by running [0, mid) inline and [mid, d) on another core via
   llm_matmul_rows(). ---- */
typedef struct {
  float* out; const QMat* w; int bits, gs;
  const int8_t* qx; const float* sx; int n, d;
} MMJob;
extern void (*llm_parallel_matmul)(const MMJob* job);

/* row-range kernel, callable from worker cores */
void llm_matmul_rows(float* out, const QMat* w, int bits, int gs,
                     const int8_t* qx, const float* sx, int n, int r0, int r1);

/* ---- API ---- */
int  llm_init(LLM* m, const uint8_t* image, size_t image_size);
void llm_forward(LLM* m, int token, int pos);  /* embed+layers+head (full models) */

/* split-model API (two-board pipeline). The activation vector m->x (dim
 * floats) is what travels over the wire between boards. */
void llm_embed(LLM* m, int token);             /* needs LLM_HAS_EMB  */
void llm_layers(LLM* m, int pos);              /* runs this image's layers on m->x */
void llm_head(LLM* m);                         /* needs LLM_HAS_HEAD; fills logits */

/* tokenizer (reads blob from image) */
int  llm_tok_init(LLM* m);
int  llm_encode(LLM* m, const char* text, int bos, int eos, int* tokens, int max);
const char* llm_decode(LLM* m, int prev_token, int token);

/* sampling */
typedef struct { float temperature, topp; uint64_t rng; } Sampler;
int llm_sample(LLM* m, Sampler* s);

#ifdef __cplusplus
}
#endif
#endif
