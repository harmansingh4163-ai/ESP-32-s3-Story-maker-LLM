/*
 * esp32_rag_storyteller.ino — RAG on a microcontroller
 * -----------------------------------------------------
 * Merges the two projects: instant fact retrieval (knowledge-bot engine,
 * corpus on SD card) + the on-device LLM (Stories15M from flash), which
 * weaves the retrieved fact into a story. The model doesn't need to KNOW
 * anything — retrieval supplies the truth, the LLM supplies the prose.
 *
 * Sketch folder must contain: this file, llm_core.c/.h, rag.c/.h,
 * partitions.csv. Model flashed at 0x1F0000 (see export_model.py),
 * corpus.bin (from make_corpus.py) at the root of the SD card.
 *
 * SD pins below default to common ESP32-S3 FSPI wiring — CHECK YOUR BOARD:
 *   Waveshare ESP32-S3-Touch-LCD-5 and Guition JC3248W535C each route the
 *   TF slot differently; consult the board schematic and edit the defines.
 *
 * Serial @115200:
 *   ask a question        -> fact + a story woven around it
 *   /fact                 -> toggle fact-only mode (instant, no LLM)
 *   /temp /topp /len /stats
 */
#include <Arduino.h>
#include <SPI.h>
#include <SD.h>
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
#include "rag.h"

/* ---- SD pins: EDIT FOR YOUR BOARD ---- */
#define SD_CS    10
#define SD_MOSI  11
#define SD_SCK   12
#define SD_MISO  13

static float g_temp = 0.8f, g_topp = 0.9f;
static int   g_maxlen = 140;
static bool  g_fact_only = false;
static LLM g_m = {};
static Sampler g_sampler = { 0.8f, 0.9f, 0 };
static bool g_llm_ready = false, g_rag_ready = false;

static void* alloc_big(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
  return p ? p : heap_caps_malloc(n, MALLOC_CAP_8BIT);
}
static void* alloc_small(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  return p ? p : heap_caps_malloc(n, MALLOC_CAP_8BIT);
}

/* dual-core matmul (same as storyteller) */
static TaskHandle_t s_worker = nullptr, s_main = nullptr;
static volatile const MMJob* s_job = nullptr;
static void worker_task(void*) {
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
  s_job = j;
  xTaskNotifyGive(s_worker);
  llm_matmul_rows(j->out, j->w, j->bits, j->gs, j->qx, j->sx, j->n, 0, j->d / 2);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static bool load_model() {
  const esp_partition_t* part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("no 'model' partition"); return false; }
  const void* base = nullptr; esp_partition_mmap_handle_t h;
  if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA,
                         &base, &h) != ESP_OK) { Serial.println("mmap failed"); return false; }
  if (llm_init(&g_m, (const uint8_t*)base, part->size)) { Serial.println("llm_init failed — model flashed at 0x1F0000?"); return false; }
  if (llm_tok_init(&g_m)) { Serial.println("tokenizer failed"); return false; }
  Serial.printf("LLM ready: %dM-class, INT%d\n",
                (g_m.h.dim * g_m.h.dim * 4 * g_m.h.n_layers / 1000000) + 9,
                g_m.h.bits);
  return true;
}

static bool load_corpus() {
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS)) { Serial.println("SD init failed — check pins/format (FAT32)"); return false; }
  File f = SD.open("/corpus.bin");
  if (!f) { Serial.println("corpus.bin not found on SD root"); return false; }
  size_t sz = f.size();
  uint8_t* buf = (uint8_t*)alloc_big(sz);
  if (!buf || f.read(buf, sz) != (int)sz) { Serial.println("corpus read failed"); return false; }
  f.close();
  int n = rag_load(buf, sz);
  if (n <= 0) { Serial.printf("corpus parse failed (%d)\n", n); return false; }
  Serial.printf("Corpus loaded: %d passages (%u KB in PSRAM)\n", n, sz / 1024);
  return true;
}

static void weave_story(const char* fact) {
  static char prompt[768];
  static int tokens[600];
  snprintf(prompt, sizeof(prompt),
           "Once upon a time, a wise old owl told the children a secret: %s "
           "The children", fact);
  int n = llm_encode(&g_m, prompt, 1, 0, tokens, 600);
  if (n >= g_m.h.seq_len - 16) { Serial.println("(fact too long for story mode)"); return; }
  g_sampler.temperature = g_temp; g_sampler.topp = g_topp;
  Serial.println("\n--- story ---");
  Serial.print(prompt);
  uint32_t t0 = millis(); int gen = 0;
  int tok = tokens[0];
  int total = min(n + g_maxlen, (int)g_m.h.seq_len);
  for (int pos = 0; pos < total; pos++) {
    llm_forward(&g_m, tok, pos);
    int next = (pos < n - 1) ? tokens[pos + 1] : llm_sample(&g_m, &g_sampler);
    if (next == 1 || next == 2) break;
    if (pos >= n - 1) { Serial.print(llm_decode(&g_m, tok, next)); gen++; }
    tok = next;
  }
  Serial.printf("\n[%.2f tok/s]\n", gen * 1000.0f / (millis() - t0));
}

static void answer(const char* q) {
  int idx[3], sc[3], cov;
  int k = rag_query_cov(q, idx, sc, 3, &cov);
  if (k == 0 || sc[0] < 20 || cov < 40) {
    Serial.println("\nI don't have anything about that in my corpus.");
    return;
  }
  const RagPassage* p = rag_get(idx[0]);
  Serial.printf("\n[FACT] %s\n", p->text);
  if (k > 1 && sc[1] >= sc[0] / 2)
    Serial.printf("[ALSO] %.80s...\n", rag_get(idx[1])->text);
  if (!g_fact_only && g_llm_ready) weave_story(p->text);
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n===========================================");
  Serial.println("  ESP32-S3 RAG Storyteller");
  Serial.println("  SD corpus retrieval + on-device LLM");
  Serial.println("===========================================");
  if (!psramFound()) { Serial.println("ERROR: enable PSRAM in Tools menu"); return; }

  s_main = xTaskGetCurrentTaskHandle();
  xTaskCreatePinnedToCore(worker_task, "mm", 4096, nullptr,
                          configMAX_PRIORITIES - 2, &s_worker, 0);
  llm_alloc_big = alloc_big; llm_alloc_small = alloc_small;
  llm_parallel_matmul = parallel_matmul;

  g_rag_ready = load_corpus();
  g_llm_ready = load_model();
  g_sampler.rng = esp_random() | 1ULL;
  Serial.printf("\nRAG:%s LLM:%s — ask me something!\n",
                g_rag_ready ? "ok" : "OFF", g_llm_ready ? "ok" : "OFF");
  Serial.println("(/fact toggles instant fact-only mode)\n");
}

void loop() {
  static char line[256]; static int pos = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (!pos) continue;
      line[pos] = '\0'; pos = 0;
      float fv; int iv;
      if (!strcmp(line, "/fact")) {
        g_fact_only = !g_fact_only;
        Serial.printf("fact-only: %s\n", g_fact_only ? "ON" : "OFF");
      }
      else if (sscanf(line, "/temp %f", &fv) == 1) g_temp = fv;
      else if (sscanf(line, "/topp %f", &fv) == 1) g_topp = fv;
      else if (sscanf(line, "/len %d", &iv) == 1) g_maxlen = iv;
      else if (!strcmp(line, "/stats"))
        Serial.printf("PSRAM free %u KB, internal %u KB\n",
                      heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1024,
                      heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024);
      else if (g_rag_ready) answer(line);
      else Serial.println("corpus not loaded");
    } else if (pos < 255) line[pos++] = c;
  }
}
