# ESP32 AI Suite — START HERE

Four projects, easiest to hardest. Each numbered folder is a complete
Arduino sketch folder: open the .ino inside it and everything it needs is
already next to it.

```
1_knowledge_bot/        instant offline Q&A bot          (easiest, no extras)
2_storyteller/          real LLM writing stories         (needs model on flash)
3_rag_storyteller/      facts from SD + LLM stories      (needs model + SD card)
4_two_board_pipeline/   one LLM split across two boards  (hardest, needs wiring)
pc_tools/               scripts you run on your PC
verification_tests/     proof everything works (optional reading)
```

---

## ONE-TIME SETUP (PC)

1. Install **Arduino IDE** + the **esp32 board package** (Boards Manager →
   search "esp32" by Espressif).
2. Install **Python 3** with numpy (`pip install numpy`).
3. Download the model files (only needed for projects 2-4):
   - https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
   - https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
   Put both in `pc_tools/`.

Arduino settings for ALL projects (Tools menu):
- Board: **ESP32S3 Dev Module**
- PSRAM: **OPI PSRAM** (try QSPI if boot fails)
- Flash Size: **16MB**
- The partitions.csv in each sketch folder is picked up automatically;
  if your IDE ignores it, choose Partition Scheme: Custom.

---

## PROJECT 1 — Knowledge Bot (5 minutes)

No model, no SD card. Open `1_knowledge_bot/esp32_knowledge_bot.ino`,
upload, open Serial Monitor at **115200**, ask questions. Edit the `KB[]`
array in the sketch to teach it your own facts.

---

## PROJECT 2 — Storyteller LLM (30 minutes)

**On PC** (inside `pc_tools/`):
```bash
python3 export_model.py stories15M.bin tokenizer.bin model_esp.bin --bits 4
```

**On board:**
1. Open `2_storyteller/esp32_storyteller.ino`, upload.
2. Flash the model to its partition (board still plugged in, close Serial
   Monitor first):
```bash
esptool.py --chip esp32s3 --port YOUR_PORT write_flash 0x1F0000 model_esp.bin
```
   (YOUR_PORT = COM5 on Windows, /dev/ttyACM0 on Linux — same port Arduino
   uses. esptool comes with the esp32 package, or `pip install esptool`.)
3. Serial Monitor @115200 → type "Once upon a time there was a brave cat"
   → watch it write. Expect ~0.5–2 tokens/sec.

---

## PROJECT 3 — RAG Storyteller (45 minutes)

Everything from Project 2, plus:

**On PC:**
```bash
python3 make_corpus.py --starter corpus.bin     # or: make_corpus.py myfacts.txt corpus.bin
```
Copy `corpus.bin` to the root of a **FAT32-formatted** SD card, insert into
the board.

**Before uploading:** open `3_rag_storyteller/esp32_rag_storyteller.ino`
and set the four `SD_*` pin defines to match YOUR board's TF slot — the
Waveshare 5" and the JC3248W535C are wired differently (check each board's
schematic/wiki). Then upload (model already flashed from Project 2).

Ask "how cold does it get in fort mcmurray" → instant [FACT] + a story.
`/fact` toggles instant-answer-only mode.

To use your own knowledge: write one fact per line in a .txt file, run
make_corpus.py on it, replace corpus.bin on the SD card. That's it —
no retraining, no reflashing.

---

## PROJECT 4 — Two-Board Pipeline (weekend project)

**On PC:**
```bash
python3 export_model.py stories15M.bin tokenizer.bin full.bin --bits 4
python3 split_image.py full.bin 3 worker.bin head.bin
```

**Wiring** (3 jumper wires between the boards):
```
head GPIO17 (TX) ──→ worker GPIO18 (RX)
head GPIO18 (RX) ←── worker GPIO17 (TX)
head GND        ───  worker GND
```
Keep wires short (<15 cm). Pins are #defines in pipeline_link.h if 17/18
clash with something on your board.

**Boards:**
- Worker board (JC3248W535C): upload `pipeline_worker/pipeline_worker.ino`,
  then `esptool.py write_flash 0x1F0000 worker.bin`
- Head board (Waveshare): upload `pipeline_head/pipeline_head.ino`,
  then `esptool.py write_flash 0x1F0000 head.bin`

Power both, Serial Monitor on the **head** board, type a prompt. If you see
"link timeout" → check TX/RX aren't swapped and GND is connected. CRC
retries → lower LINK_BAUD to 921600 in pipeline_link.h on BOTH boards.

Once 15M works split, the real prize: **stories42M.bin** (same HuggingFace
page) is too big for one board but fits across two. Export with --bits 4
--gs 32, split at layer 4.

---

## If something fails

| Symptom | Fix |
|---|---|
| "no 'model' partition" | partitions.csv wasn't used — set Partition Scheme: Custom, re-upload |
| "llm_init failed" | model not flashed / wrong offset — redo the esptool command |
| "PSRAM not found" | Tools → PSRAM → OPI (or QSPI), re-upload |
| SD init failed | wrong pins (edit defines) or card not FAT32 |
| link timeout | TX/RX swapped, or GND missing |
| garbled serial text | Serial Monitor not at 115200 |

All the math was verified on a PC before you got it (see
verification_tests/) — so failures will be wiring, pins, or flashing steps,
not the AI itself.
