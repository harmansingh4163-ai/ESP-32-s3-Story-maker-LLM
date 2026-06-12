/* Chain test: head.embed -> worker.layers -> head.layers -> head.head
 * must equal the full model's logits exactly (same weights, same math). */
#include "llm_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint8_t* slurp(const char* p, long* sz) {
  FILE* f = fopen(p,"rb"); if(!f){perror(p);exit(1);}
  fseek(f,0,SEEK_END); *sz=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b = malloc(*sz);
  if (fread(b,1,*sz,f)!=(size_t)*sz) exit(1);
  fclose(f); return b;
}

int main(void) {
  long s1,s2,s3;
  uint8_t* full_img = slurp("synth_q4.bin",&s1);
  uint8_t* w_img    = slurp("worker_q4.bin",&s2);
  uint8_t* h_img    = slurp("head_q4.bin",&s3);

  LLM full={0}, A={0}, B={0};
  int r;
  if ((r=llm_init(&full, full_img, s1))) { printf("full init %d\n",r); return 1; }
  if ((r=llm_init(&A, w_img, s2)))       { printf("worker init %d\n",r); return 1; }
  if ((r=llm_init(&B, h_img, s3)))       { printf("head init %d\n",r); return 1; }
  printf("full L=%d | worker local=%d flags=%d | head local=%d flags=%d\n",
         full.h.n_layers, A.h.local_layers, A.h.flags,
         B.h.local_layers, B.h.flags);

  int tokens[] = {5, 17, 42, 9, 100, 3};
  double worst = 0;
  for (int pos = 0; pos < 6; pos++) {
    llm_forward(&full, tokens[pos], pos);
    /* pipeline: B embeds, "wire" to A, A's layers, "wire" back, B finishes */
    llm_embed(&B, tokens[pos]);
    memcpy(A.x, B.x, full.h.dim * 4);          /* UART send */
    llm_layers(&A, pos);
    memcpy(B.x, A.x, full.h.dim * 4);          /* UART recv */
    llm_layers(&B, pos);
    llm_head(&B);
    double md = 0;
    for (int i = 0; i < full.h.vocab_size; i++) {
      double d = fabs(full.logits[i] - B.logits[i]);
      if (d > md) md = d;
    }
    if (md > worst) worst = md;
    printf("  pos %d: max abs logit diff %.2e\n", pos, md);
  }
  printf("pipeline vs monolithic: worst %.2e -> %s\n", worst,
         worst < 1e-5 ? "PASS" : "FAIL");
  return worst >= 1e-5;
}
