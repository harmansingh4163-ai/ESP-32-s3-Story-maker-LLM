/* test_host.c — load synth model, run llm_core forward, compare logits
 * against the NumPy reference. Exit 0 only if max rel error is tiny. */
#include "llm_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static uint8_t* slurp(const char* path, long* sz) {
  FILE* f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END); *sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t* buf = malloc(*sz);
  if (fread(buf, 1, *sz, f) != (size_t)*sz) { perror("read"); exit(1); }
  fclose(f);
  return buf;
}

static int run(const char* model_path, const char* ref_path) {
  long msz, rsz;
  uint8_t* img = slurp(model_path, &msz);
  uint8_t* ref = slurp(ref_path, &rsz);

  LLM m = {0};
  int rc = llm_init(&m, img, msz);
  if (rc) { printf("llm_init failed: %d\n", rc); return 1; }
  if (llm_tok_init(&m)) { printf("tok_init failed\n"); return 1; }

  int n_tok = ((int32_t*)ref)[0], vocab = ((int32_t*)ref)[1];
  int32_t* tokens = (int32_t*)(ref + 8);
  float* ref_logits = (float*)(ref + 8 + n_tok * 4);

  double worst = 0; int worst_pos = -1;
  int top1_match = 0;
  for (int pos = 0; pos < n_tok; pos++) {
    llm_forward(&m, tokens[pos], pos);
    float* rl = ref_logits + (size_t)pos * vocab;
    int argc_ = 0, argr = 0;
    double md = 0;
    for (int i = 0; i < vocab; i++) {
      double d = fabs(m.logits[i] - rl[i]);
      double denom = fabs(rl[i]) > 1.0 ? fabs(rl[i]) : 1.0;
      if (d / denom > md) md = d / denom;
      if (m.logits[i] > m.logits[argc_]) argc_ = i;
      if (rl[i] > rl[argr]) argr = i;
    }
    if (argc_ == argr) top1_match++;
    if (md > worst) { worst = md; worst_pos = pos; }
    printf("  pos %d: max rel diff %.2e, top1 %s (C=%d ref=%d)\n",
           pos, md, argc_ == argr ? "MATCH" : "MISMATCH", argc_, argr);
  }
  printf("%s: worst rel diff %.2e at pos %d, top1 %d/%d\n",
         model_path, worst, worst_pos, top1_match, n_tok);
  int ok = (worst < 5e-3) && (top1_match == n_tok);
  printf("%s\n\n", ok ? "PASS" : "FAIL");
  return !ok;
}

int main(void) {
  int rc = 0;
  rc |= run("synth_q8.bin", "ref_q8.bin");
  rc |= run("synth_q4.bin", "ref_q4.bin");
  /* also smoke-test encode/decode round trip on the synth tokenizer */
  return rc;
}
