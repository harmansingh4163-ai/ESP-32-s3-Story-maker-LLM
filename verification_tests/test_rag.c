#include "rag.h"
#include <stdio.h>
#include <stdlib.h>
int main(void) {
  FILE* f = fopen("corpus.bin","rb");
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  unsigned char* buf = malloc(sz);
  if (fread(buf,1,sz,f)!=(size_t)sz) return 1;
  fclose(f);
  int n = rag_load(buf, sz);
  printf("loaded %d passages\n\n", n);
  const char* queries[] = {
    "how many harts does an octopus have",      // typo: harts
    "why is honey special",
    "tell me about the moon",
    "fastest animal",
    "esp32 chip",
    "how cold does it get in fort mcmurray",
    "do penguins fly",
    "what is the biggest animal ever",
    "quantum mechanics of black holes",          // not in corpus
  };
  for (size_t i = 0; i < sizeof(queries)/sizeof(*queries); i++) {
    int idx[3], sc[3];
    int cov; int k = rag_query_cov(queries[i], idx, sc, 3, &cov);
    printf("Q: %s\n", queries[i]);
    if (k == 0) printf("   (no match)\n");
    else printf("   [score %d cov %d%%] %.60s...\n", sc[0], cov, rag_get(idx[0])->text);
    printf("\n");
  }
  return 0;
}
