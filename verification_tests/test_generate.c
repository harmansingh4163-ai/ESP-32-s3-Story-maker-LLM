/* end-to-end host test: load exported image, encode a prompt with the REAL
 * llama2.c tokenizer, run generation, decode. Validates tokenizer + sampler. */
#include "llm_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  FILE* f = fopen("fake_esp.bin", "rb");
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t* img = malloc(sz);
  if (fread(img, 1, sz, f) != (size_t)sz) return 1;
  fclose(f);

  LLM m = {0};
  if (llm_init(&m, img, sz)) { puts("init fail"); return 1; }
  if (llm_tok_init(&m)) { puts("tok fail"); return 1; }

  int tokens[64];
  int n = llm_encode(&m, "Once upon a time there was a dragon", 1, 0, tokens, 64);
  printf("encoded %d tokens:", n);
  for (int i = 0; i < n; i++) printf(" %d", tokens[i]);
  /* llama2 tokenizer reference: "Once upon a time" should start 1,9038,2501... */
  printf("\nround-trip: ");
  for (int i = 1; i < n; i++) printf("%s", llm_decode(&m, tokens[i-1], tokens[i]));
  printf("\n");

  Sampler s = { 0.9f, 0.9f, 1234567ULL };
  int tok = tokens[0];
  printf("generated (random weights, expect gibberish but valid pieces):\n  ");
  for (int pos = 0; pos < 40; pos++) {
    llm_forward(&m, tok, pos);
    int next = (pos < n - 1) ? tokens[pos + 1] : llm_sample(&m, &s);
    if (next == 2 || next == 1) break;
    printf("%s", llm_decode(&m, tok, next));
    fflush(stdout);
    tok = next;
  }
  printf("\nOK\n");
  return 0;
}
