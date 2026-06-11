## Hardware

Verified on (this exact pair runs the pipeline in the photo):
| Board | Role | Notes |
|---|---|---|
| Waveshare ESP32-S3-Touch-LCD-5 | head | 16MB flash, 8MB OPI PSRAM. No free GPIO header — link wired via the I2C terminal block (GPIO8=TX, GPIO9=SCL=RX) and Sensor-AD GND. Set LINK_RX_PIN 9 / LINK_TX_PIN 8. |
| Guition JC3248W535C (3.5" 320×480) | worker | 16MB flash, 8MB PSRAM. GPIO 17/18 exposed on P3/P4 connectors — default pins work. |

Should work on any ESP32-S3 with **16MB flash + 8MB PSRAM** (the common 
N16R8 spec): generic S3 DevKitC-1 N16R8 boards, most S3 display boards. 
Single-board projects (storyteller, RAG) need one such board; the pipeline 
needs two — they don't have to be the same model.

Won't work: original ESP32 / ESP32-S2 / C3 (no/insufficient PSRAM, smaller 
flash), S3 variants with 8MB flash or no PSRAM. The plain ESP32 *can* run 
the knowledge_bot project (no model needed).

Displays unused in v1 — all interaction over USB serial. Both boards' 
screens stay dark by design; touchscreen UI is on the roadmap.



# Full Setup Guide (Windows)

## 0. One-time PC setup
winget install Python.Python.3.12        # then CLOSE and reopen cmd
python --version                          # must print a version
pip install numpy esptool

If "Python was not found" after install: Windows Settings → 
"Manage app execution aliases" → turn OFF python.exe and python3.exe.

## 1. Get the model files (into pc_tools/)
Browser-download these two:
- https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
- https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
  (right-click → Save link as → must save as tokenizer.bin, not .txt)

## 2. Convert the model
cd path\to\repo\pc_tools
python export_model.py stories15M.bin tokenizer.bin full.bin --bits 4

## 3. Split for two boards
python split_image.py full.bin 3 worker.bin head.bin

## 4. Arduino upload (each board)
Sketch folders must be self-contained: copy core/* and partitions.csv 
into the sketch folder first.
Tools settings: ESP32S3 Dev Module · USB CDC On Boot: Enabled · 
Flash 16MB · PSRAM: OPI · 240MHz
Upload pipeline_head to board A, pipeline_worker to board B.
If Serial Monitor shows the boot log but no banner: flip USB CDC 
On Boot, re-upload.

## 5. Flash the model halves (SERIAL MONITOR MUST BE CLOSED)
python -m esptool --chip esp32s3 --port COM4 write_flash 0x1F0000 head.bin
python -m esptool --chip esp32s3 --port COM7 write_flash 0x1F0000 worker.bin
(find your COM numbers in Arduino: Tools → Port)
Success looks like: "Hash of data verified."

## 6. Wiring (3 jumper wires, boards powered off)
head TX-pin  → worker RX-pin     |  defaults: 17→18
head RX-pin  ← worker TX-pin     |  defaults: 18←17
GND          — GND               |  any GND pin on each board
Pins are #defines in pipeline_link.h (per board — they don't have to match
the other board, only the wiring). Baud must match on BOTH boards.

## 7. Run
Serial Monitor on the HEAD board, 115200, reset both boards,
wait for "Ready:", type a story opening.

## Troubleshooting
| Symptom | Fix |
|---|---|
| llm_init -1 | model not flashed / wrong offset — redo step 5 |
| no 'model' partition | partitions.csv not used — Partition Scheme: Custom |
| link timeout | TX/RX swapped, or GND wire missing |
| crc error, retry (constant) | lower LINK_BAUD on both boards |
| boot log but no banner | USB CDC On Boot toggle + re-upload |
| esptool "No such file" | wrong cmd folder — cd to where the .bin is |



## Upgrading to stories42M (two boards required)

> Status: export path implemented and size-budgeted; the 15M pipeline is 
> hardware-verified. 42M is the designed payload — measured numbers welcome 
> via issues/PRs.

The 42M model (~24MB at INT4) cannot fit a single 16MB board — this is 
the configuration the pipeline exists for. Same wiring, same sketches, 
no code changes: only the model images differ.

### 1. Get the model (~170MB download)
https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin

### 2. Export — note BOTH extra flags
python export_model.py stories42M.bin tokenizer.bin full42.bin --bits 4 --gs 32 --seq 224

--seq 224 is REQUIRED, not optional: 42M was trained with a 1024-token 
context, whose KV cache (~29MB) would exceed the 8MB PSRAM. Capping at 
224 fits (~3.7MB) and still allows full short stories.

### 3. Split at layer 7 (not 3)
python split_image.py full42.bin 7 worker42.bin head42.bin

42M's embedding table (~9MB) lives on the head, so the split is lopsided: 
worker takes layers 0–6, head takes embedding + layer 7 + classifier.
CHECK the printed sizes: each image must be under 14.6MB. If worker42.bin 
is over, your group size is wrong — re-export with --gs 32.

### 4. Flash (larger files — ~10 min each at typical USB speeds)
python -m esptool --chip esp32s3 --port COM_head   write_flash 0x1F0000 head42.bin
python -m esptool --chip esp32s3 --port COM_worker write_flash 0x1F0000 worker42.bin

### 5. Verify at boot
Head banner should read "emb + 1 local layers of 8 total"; worker 
"7 local layers, dim 512". Then prompt as usual.

### Expectations
~0.4–0.7 tok/s (vs ~1.4 for 15M) — double the weight streamed per token 
through the same flash bandwidth. In exchange: markedly better coherence; 
TinyStories models in this range outperform GPT-2 XL (100× larger) on 
simple-story quality (Eldan & Li, 2023).

If the worker hits an allocation failure at boot: re-export with --seq 192.
