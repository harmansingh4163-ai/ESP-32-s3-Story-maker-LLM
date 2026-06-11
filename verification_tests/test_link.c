#include "pipeline_link.h"
#include "llm_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint8_t wire[65536]; static size_t w_head = 0, w_tail = 0;
static void w_write(const uint8_t* d, size_t n) {
  memcpy(wire + w_head, d, n); w_head += n;
}
static int w_read(uint32_t to) {
  (void)to;
  return (w_tail < w_head) ? wire[w_tail++] : -1;
}
static const LinkIO IO = { w_write, w_read };

static uint8_t* slurp(const char* p, long* sz) {
  FILE* f = fopen(p,"rb"); fseek(f,0,SEEK_END); *sz=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t* b = malloc(*sz);
  if (fread(b,1,*sz,f)!=(size_t)*sz) exit(1);
  fclose(f); return b;
}

int main(void) {
  long s1,s2,s3;
  uint8_t* full_img = slurp("synth_q4.bin",&s1);
  uint8_t* w_img = slurp("worker_q4.bin",&s2);
  uint8_t* h_img = slurp("head_q4.bin",&s3);
  LLM full={0}, A={0}, B={0};
  llm_init(&full, full_img, s1); llm_init(&A, w_img, s2); llm_init(&B, h_img, s3);
  int dim = full.h.dim;

  /* simulate the full head<->worker exchange THROUGH the framed protocol,
     with line noise injected before each frame */
  static uint8_t buf[4096];
  int tokens[] = {5,17,42,9,100,3};
  double worst = 0;
  for (int pos = 0; pos < 6; pos++) {
    llm_forward(&full, tokens[pos], pos);

    /* HEAD side: embed, frame up pos+x, send */
    llm_embed(&B, tokens[pos]);
    uint16_t plen = 2 + dim * 4;
    buf[0] = pos & 0xFF; buf[1] = pos >> 8;
    memcpy(buf + 2, B.x, dim * 4);
    uint8_t noise[3] = {0xFF, 0xA5, 0x00};       /* garbage before frame */
    w_write(noise, 3);
    link_send(&IO, CMD_FWD, buf, plen);

    /* WORKER side: receive, run layers, reply */
    uint16_t got;
    int cmd = link_recv(&IO, buf, sizeof(buf), &got, 100);
    if (cmd != CMD_FWD || got != plen) { printf("FWD recv fail %d\n", cmd); return 1; }
    int rpos = buf[0] | (buf[1] << 8);
    memcpy(A.x, buf + 2, dim * 4);
    llm_layers(&A, rpos);
    link_send(&IO, CMD_RSP, (uint8_t*)A.x, dim * 4);

    /* HEAD side: receive, finish */
    cmd = link_recv(&IO, buf, sizeof(buf), &got, 100);
    if (cmd != CMD_RSP || got != dim * 4) { printf("RSP recv fail\n"); return 1; }
    memcpy(B.x, buf, dim * 4);
    llm_layers(&B, pos);
    llm_head(&B);

    for (int i = 0; i < full.h.vocab_size; i++) {
      double d = fabs(full.logits[i] - B.logits[i]);
      if (d > worst) worst = d;
    }
  }
  printf("over-the-wire pipeline vs monolithic: worst diff %.2e -> %s\n",
         worst, worst < 1e-5 ? "PASS" : "FAIL");

  /* corrupted frame must be rejected by CRC */
  link_send(&IO, CMD_FWD, (uint8_t*)"hello", 5);
  wire[w_head - 4] ^= 0x42;                       /* flip a payload bit */
  uint16_t got;
  int cmd = link_recv(&IO, buf, sizeof(buf), &got, 100);
  printf("corrupted frame -> %s\n", cmd == -2 ? "rejected (PASS)" : "ACCEPTED (FAIL)");
  return !(worst < 1e-5 && cmd == -2);
}
