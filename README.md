# esp32-llm-pipeline

**One language model, two microcontrollers.** A Llama-architecture LLM
running with its layers split across two ESP32-S3 boards — per token, the
activation vector crosses three wires (CRC-framed UART) between the chips.

![two boards wired together](photo.jpg)

```
> Once upon a time there was a brave little fox
cat and fish are friends. They like to play and swim in the pond.
One day, the cat sees a bird on a tree...
[1.4 tok/s across 2 boards]
```

As far as published projects show, this is the **first multi-chip pipelined
LLM inference on ESP32-class hardware.**

## Why

A 16MB ESP32-S3 caps out at a ~15M-parameter model (INT4). The next
TinyStories size — stories42M, ~24MB — fits **no single board**. Splitting
layers across two chips makes combined flash the limit, not the chip.

## How it works

- **Weights:** INT4 (group-32 scales), streamed from a memory-mapped flash
  partition — **0 bytes of RAM** used for weights
- **Compute:** INT8 activations, integer-exact group dot products, matmul
  rows split across both LX7 cores of each chip
- **Split:** worker board runs layers 0–K; head board holds embedding,
  layers K–L, classifier, tokenizer, and samples each next token
- **Link:** UART @460800, frame = `A5 5A | cmd | len | payload | CRC16`,
  ~1.2KB/token round trip (~3% of token time)

## Verified, not vibes

Everything testable without hardware was tested before flashing (see /tests):
forward pass matches a NumPy reference to **~3e-7** (INT4 and INT8); the
split pipeline is **bit-exact** vs the monolithic model; the link protocol
was fuzzed — noise ignored, corrupted frames rejected by CRC.

## vs prior art

| | params | weights live in | speed | output |
|---|---|---|---|---|
| known ESP32 LLM ports | 260K | RAM | 19–33 tok/s | sentence-level babble |
| this, single board | 15M | flash (mmap) | ~1.4 tok/s | multi-paragraph stories |
| this, two boards | 15M now / 42M target | split flash | ~1.4 / est. 0.5 tok/s | better still |

Coherence isn't gradual: TinyStories research shows plot consistency
*emerges* in the millions-of-parameters range
([Eldan & Li 2023](https://arxiv.org/abs/2305.07759)).

---

# Usage

## 0. Requirements

**Hardware:** one ESP32-S3 with **16MB flash + 8MB PSRAM** for single-board
mode; two of them + 3 jumper wires for the pipeline. Verified on: Waveshare
ESP32-S3-Touch-LCD-5 (head) and Guition JC3248W535C (worker). The two
boards do not have to be the same model. Displays are unused in v1 — all
interaction is over USB serial, screens stay dark by design.

**PC (one-time setup, Windows commands shown):**
```bat
winget install Python.Python.3.12
:: close cmd, open a NEW one, then:
python --version
pip install numpy esptool
```
Arduino IDE with the esp32 board package (Boards Manager → "esp32" by
Espressif; v3.x recommended, v2.x works — the sketches include a compat shim).

> "Python was not found" after installing? Windows Settings → "Manage app
> execution aliases" → turn OFF python.exe and python3.exe, reopen cmd.

## 1. Get and convert the model

Download into `pc_tools/`:
- https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin (~60MB)
- https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin
  (right-click → Save link as → keep the name `tokenizer.bin`)

```bat
cd path\to\repo\pc_tools
python export_model.py stories15M.bin tokenizer.bin full.bin --bits 4
```
Output: `full.bin` (~10MB) — the INT4 flash image.

## 2A. Single-board storyteller (start here, ~15 min)

1. Copy `core\llm_core.c`, `core\llm_core.h`, and `partitions.csv` into
   `sketches\storyteller\`.
2. Open `esp32_storyteller.ino` in Arduino IDE. Tools settings:
   **ESP32S3 Dev Module · USB CDC On Boot: Enabled · Flash Size: 16MB ·
   PSRAM: OPI · CPU 240MHz**. Upload.
3. Flash the model (close Serial Monitor first; COMx is in Tools → Port):
```bat
python -m esptool --chip esp32s3 --port COMx write_flash 0x1F0000 full.bin
```
   Success = "Hash of data verified."
4. Serial Monitor @ **115200**, press the board's reset button, wait for
   `Ready`, type a story opening, press Enter.

## 2B. Two-board pipeline

**Split the model** (worker gets layers 0–2; head gets embedding + 3–5):
```bat
python split_image.py full.bin 3 worker.bin head.bin
```

**Firmware:** copy `core\*` and `partitions.csv` into BOTH
`sketches\pipeline_head\` and `sketches\pipeline_worker\`. Upload
`pipeline_head.ino` to board A and `pipeline_worker.ino` to board B
(same Tools settings as 2A).

**Pins & baud** — edit each sketch's copy of `core/pipeline_link.h` if
needed. Each board's `LINK_TX_PIN`/`LINK_RX_PIN` must match *its own
wiring*; the two boards' pin numbers do NOT have to match each other.
`LINK_BAUD` MUST be identical on both. Defaults: 17/18 @460800. On the
Waveshare 5" (no free GPIO header) use the I2C terminal block:
TX = GPIO8 (SDA), RX = GPIO9 (SCL).

**Flash each half** (each board's own COM port, Serial Monitor closed):
```bat
python -m esptool --chip esp32s3 --port COM_head   write_flash 0x1F0000 head.bin
python -m esptool --chip esp32s3 --port COM_worker write_flash 0x1F0000 worker.bin
```

**Wire — 3 jumpers, boards powered off, TX↔RX crossed:**
```
head TX-pin ──→ worker RX-pin
head RX-pin ←── worker TX-pin
GND ─────────── GND     (any GND pin on each board; do NOT connect 3V3/VCC)
```

**Run:** power both (head on the PC; worker on any USB power). Serial
Monitor on the HEAD's port @115200, reset both boards, wait for
`Ready: emb + 3 local layers of 6 total`, type a prompt.

## 3. Runtime commands (typed into Serial Monitor)

| Command | Effect |
|---|---|
| any text | generates a story continuing your text |
| `/temp 0.7` | lower = focused, higher (1.0+) = wilder |
| `/topp 0.9` | nucleus sampling cutoff |
| `/len 250` | max tokens per prompt |
| `/stats` | free RAM/PSRAM and current settings |

## 4. Upgrading to stories42M (the model that needs two boards)

> Status: export path implemented and size-budgeted; the 15M pipeline is
> hardware-verified. Measured 42M numbers welcome via issues.

```bat
:: download stories42M.bin (~170MB) from the same HuggingFace page, then:
python export_model.py stories42M.bin tokenizer.bin full42.bin --bits 4 --gs 32 --seq 224
python split_image.py full42.bin 7 worker42.bin head42.bin
python -m esptool --chip esp32s3 --port COM_head   write_flash 0x1F0000 head42.bin
python -m esptool --chip esp32s3 --port COM_worker write_flash 0x1F0000 worker42.bin
```
`--seq 224` is **required** (42M's native 1024-token context would need a
~29MB KV cache — over the 8MB PSRAM; 224 fits). Split at **7**, not 3:
the ~9MB embedding lives on the head, so layers skew to the worker. Check
the printed image sizes stay under 14.6MB. Boot banners should read
"emb + 1 local layers of 8" (head) and "7 local layers, dim 512" (worker).
Expect ~0.4–0.7 tok/s — and clearly better stories. Worker allocation
failure at boot → re-export with `--seq 192`.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `llm_init -1` | no/old model in flash — redo the esptool step at 0x1F0000 |
| `no 'model' partition` | partitions.csv not applied — copy into the sketch folder, set Partition Scheme: Custom, re-upload |
| boot log but no banner | flip Tools → USB CDC On Boot, re-upload |
| `PSRAM not found` | Tools → PSRAM: OPI (or QSPI), re-upload |
| `link timeout` | TX/RX swapped at one end, or GND wire missing |
| `crc error, retry` constantly | shorten wires or set LINK_BAUD 115200 on BOTH boards |
| esptool: No such file | cd into the folder containing the .bin |
| esptool: port busy | close Serial Monitor |

---

## Repo layout

```
core/            inference engine + link protocol (host-verified, portable C)
sketches/        Arduino firmware (copy core/* + partitions.csv in before building)
pc_tools/        model quantizer/exporter and the layer splitter
tests/           the proof: NumPy-reference, bit-exactness, and protocol fuzz tests
partitions.csv   16MB flash layout with the 14MB model partition
```

## Roadmap
- [ ] stories42M measured on hardware
- [ ] PIE SIMD in the marked matmul slot (`llm_matmul_rows`) — est. 2-3×
- [ ] On-device touch UI (no PC)

## Credits

MIT License. Engine architecture after
[llama2.c](https://github.com/karpathy/llama2.c) (Andrej Karpathy, MIT);
models trained on [TinyStories](https://arxiv.org/abs/2305.07759)
(Eldan & Li, Microsoft Research). Code developed in collaboration with
Claude (Anthropic); hardware, integration, and debugging by me.
