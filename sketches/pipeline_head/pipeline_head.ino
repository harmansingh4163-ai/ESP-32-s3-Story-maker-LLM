/*
 * pipeline_head.ino — board B: the user-facing board.
 * Holds embedding + last layers + classifier + tokenizer (head.bin).
 * Per token: embed -> ship activation to worker over UART -> worker runs
 * its layers -> finish locally -> sample. With retry on link errors.
 *
 * Sketch folder: this file + llm_core.c/.h + pipeline_link.h + partitions.csv
 */
#include <Arduino.h>
#include "esp_partition.h"
/* compat: Arduino-esp32 core 2.x (IDF4) used older names for the mmap API */
#include "esp_idf_version.h"
#if ESP_IDF_VERSION < ESP_IDF_VERSION_VAL(5, 0, 0)
  #include "esp_spi_flash.h"
  typedef spi_flash_mmap_handle_t esp_partition_mmap_handle_t;
  #define ESP_PARTITION_MMAP_DATA SPI_FLASH_MMAP_DATA
#endif
#include "esp_heap_caps.h"
#include "llm_core.h"
#include "pipeline_link.h"

static LLM g_m = {};
static Sampler g_sampler = { 0.8f, 0.9f, 0 };
static uint8_t g_buf[4096];
static float g_temp = 0.8f, g_topp = 0.9f;
static int g_maxlen = 180;
static bool g_ready = false;

static void io_write(const uint8_t* d, size_t n) { Serial1.write(d, n); }
static int io_read(uint32_t timeout_ms) {
  uint32_t t0 = millis();
  while (!Serial1.available()) {
    if (millis() - t0 >= timeout_ms) return -1;
    delayMicroseconds(50);
  }
  return Serial1.read();
}
static const LinkIO IO = { io_write, io_read };

static void* alloc_big(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  return p ? p : heap_caps_malloc(n, MALLOC_CAP_8BIT);
}

static TaskHandle_t s_worker = nullptr, s_main = nullptr;
static volatile const MMJob* s_job = nullptr;
static void mm_task(void*) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    const MMJob* j = (const MMJob*)s_job;
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx,
                    j->n, j->d / 2, j->d);
    xTaskNotifyGive(s_main);
  }
}
static void parallel_matmul(const MMJob* j) {
  if (j->d < 64) {
    llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d);
    return;
  }
  s_job = j; xTaskNotifyGive(s_worker);
  llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d / 2);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

/* one remote round-trip with one retry; returns true on success */
static bool remote_layers(int pos) {
  for (int attempt = 0; attempt < 2; attempt++) {
    uint16_t plen = 2 + g_m.h.dim * 4;
    g_buf[0] = pos & 0xFF; g_buf[1] = pos >> 8;
    memcpy(g_buf + 2, g_m.x, g_m.h.dim * 4);
    link_send(&IO, CMD_FWD, g_buf, plen);
    uint16_t got;
    int cmd = link_recv(&IO, g_buf, sizeof(g_buf), &got, 8000);
    if (cmd == CMD_RSP && got == g_m.h.dim * 4) {
      memcpy(g_m.x, g_buf, g_m.h.dim * 4);
      return true;
    }
    Serial.printf("\n[link %s, retry]", cmd == -1 ? "timeout" : "crc error");
  }
  return false;
}

static void generate(const char* prompt) {
  static int tokens[512];
  int n = llm_encode(&g_m, prompt, 1, 0, tokens, 512);
  if (n < 1 || n >= g_m.h.seq_len) { Serial.println("(bad prompt length)"); return; }
  g_sampler.temperature = g_temp; g_sampler.topp = g_topp;
  Serial.print("\n> "); Serial.print(prompt);
  uint32_t t0 = millis(); int gen = 0;
  int tok = tokens[0];
  int total = min(n + g_maxlen, (int)g_m.h.seq_len);
  for (int pos = 0; pos < total; pos++) {
    llm_embed(&g_m, tok);
    if (!remote_layers(pos)) {
      Serial.println("\nERROR: worker board not responding. Check wiring/power.");
      return;
    }
    llm_layers(&g_m, pos);
    llm_head(&g_m);
    int next = (pos < n - 1) ? tokens[pos + 1] : llm_sample(&g_m, &g_sampler);
    if (next == 1 || next == 2) break;
    if (pos >= n - 1) { Serial.print(llm_decode(&g_m, tok, next)); gen++; }
    tok = next;
  }
  Serial.printf("\n\n[%d tokens in %.1fs — %.2f tok/s across 2 boards]\n",
                gen, (millis()-t0)/1000.0f, gen * 1000.0f / (millis()-t0));
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(LINK_BAUD, SERIAL_8N1, LINK_RX_PIN, LINK_TX_PIN);
  delay(1500);
  Serial.println("\nPipeline HEAD board — two-chip LLM");
  if (!psramFound()) { Serial.println("ERROR: enable PSRAM"); return; }

  s_main = xTaskGetCurrentTaskHandle();
  xTaskCreatePinnedToCore(mm_task, "mm", 4096, nullptr,
                          configMAX_PRIORITIES - 2, &s_worker, 0);
  llm_alloc_big = alloc_big;
  llm_parallel_matmul = parallel_matmul;

  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  const void* base = nullptr; esp_partition_mmap_handle_t h;
  if (!part || esp_partition_mmap(part, 0, part->size,
        ESP_PARTITION_MMAP_DATA, &base, &h) != ESP_OK) {
    Serial.println("ERROR: model partition mmap failed"); return;
  }
  int rc = llm_init(&g_m, (const uint8_t*)base, part->size);
  if (rc) { Serial.printf("ERROR: llm_init %d — flash head.bin here\n", rc); return; }
  if (!(g_m.h.flags & LLM_HAS_EMB) || !(g_m.h.flags & LLM_HAS_HEAD)) {
    Serial.println("ERROR: this image isn't head.bin (missing emb/head)"); return;
  }
  if (llm_tok_init(&g_m)) { Serial.println("tokenizer failed"); return; }
  g_sampler.rng = esp_random() | 1ULL;
  g_ready = true;
  Serial.printf("Ready: emb + %d local layers of %d total. "
                "Type a story opening.\n", g_m.h.local_layers, g_m.h.n_layers);
}

void loop() {
  if (!g_ready) { delay(1000); return; }
  static char line[256]; static int pos = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (!pos) continue;
      line[pos] = '\0'; pos = 0;
      float fv; int iv;
      if (sscanf(line, "/temp %f", &fv) == 1) g_temp = fv;
      else if (sscanf(line, "/topp %f", &fv) == 1) g_topp = fv;
      else if (sscanf(line, "/len %d", &iv) == 1) g_maxlen = iv;
      else generate(line);
    } else if (pos < 255) line[pos++] = c;
  }
}
