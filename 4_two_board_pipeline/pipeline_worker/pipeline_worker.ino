/*
 * pipeline_worker.ino — board A: the layer coprocessor.
 * Holds the first K transformer layers (worker.bin from split_image.py,
 * flashed to its model partition). Waits for activation vectors over UART,
 * runs its layers, sends the result back. No tokenizer, no display.
 *
 * Sketch folder: this file + llm_core.c/.h + pipeline_link.h + partitions.csv
 * Wiring to head board: TX17->RX18, RX18<-TX17, GND<->GND.
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
static uint8_t g_buf[4096];

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

/* dual-core matmul, same engine as the other sketches */
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

void setup() {
  Serial.begin(115200);
  Serial1.begin(LINK_BAUD, SERIAL_8N1, LINK_RX_PIN, LINK_TX_PIN);
  delay(1000);
  Serial.println("\nPipeline WORKER board");

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
    Serial.println("ERROR: model partition mmap failed"); for(;;) delay(1000);
  }
  int rc = llm_init(&g_m, (const uint8_t*)base, part->size);
  if (rc) { Serial.printf("ERROR: llm_init %d — flash worker.bin to 0x1F0000\n", rc); for(;;) delay(1000); }
  if (g_m.h.flags != 0)
    Serial.println("WARNING: this image has emb/head — is it worker.bin?");
  Serial.printf("Ready: %d local layers, dim %d. Waiting for head board.\n",
                g_m.h.local_layers, g_m.h.dim);
}

void loop() {
  uint16_t got;
  int cmd = link_recv(&IO, g_buf, sizeof(g_buf), &got, 10000);
  if (cmd == CMD_FWD && got == 2 + g_m.h.dim * 4) {
    int pos = g_buf[0] | (g_buf[1] << 8);
    memcpy(g_m.x, g_buf + 2, g_m.h.dim * 4);
    llm_layers(&g_m, pos);
    link_send(&IO, CMD_RSP, (const uint8_t*)g_m.x, g_m.h.dim * 4);
  } else if (cmd == -2) {
    Serial.println("bad frame (crc) — head will retry");
  }
}
