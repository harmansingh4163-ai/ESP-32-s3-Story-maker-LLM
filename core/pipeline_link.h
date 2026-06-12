/* pipeline_link.h — framed UART protocol for the two-board pipeline.
 * Frame: A5 5A | cmd | len_lo | len_hi | payload[len] | crc_lo | crc_hi
 * CRC16-CCITT over cmd..payload. Byte I/O is abstracted through function
 * pointers so the framing logic is host-testable with a loopback.
 *
 * Wiring: head TX17 -> worker RX18, head RX18 <- worker TX17, GND <-> GND.
 * 2,000,000 baud: a 288-dim fp32 activation (1152 B) crosses in ~6 ms,
 * negligible against ~1 s/token inference time.
 */
#ifndef PIPELINE_LINK_H
#define PIPELINE_LINK_H
#include <stdint.h>
#include <stddef.h>

#define LINK_BAUD   2000000
#define LINK_RX_PIN 18   /* EDIT for your wiring */
#define LINK_TX_PIN 17
#define CMD_FWD 0x01     /* head -> worker: u16 pos + dim fp32 */
#define CMD_RSP 0x02     /* worker -> head: dim fp32 */

typedef struct {
  void (*write)(const uint8_t* data, size_t n);   /* blocking send */
  int  (*read)(uint32_t timeout_ms);              /* one byte or -1 */
} LinkIO;

static uint16_t link_crc_step(uint16_t c, uint8_t b) {
  c ^= (uint16_t)b << 8;
  for (int k = 0; k < 8; k++)
    c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
  return c;
}

static uint16_t link_crc(uint8_t cmd, const uint8_t* p, uint16_t len) {
  uint16_t c = 0xFFFF;
  c = link_crc_step(c, cmd);
  c = link_crc_step(c, (uint8_t)(len & 0xFF));
  c = link_crc_step(c, (uint8_t)(len >> 8));
  for (uint16_t i = 0; i < len; i++) c = link_crc_step(c, p[i]);
  return c;
}

static void link_send(const LinkIO* io, uint8_t cmd,
                      const uint8_t* payload, uint16_t len) {
  uint8_t hdr[5] = { 0xA5, 0x5A, cmd,
                     (uint8_t)(len & 0xFF), (uint8_t)(len >> 8) };
  uint16_t crc = link_crc(cmd, payload, len);
  uint8_t tail[2] = { (uint8_t)(crc & 0xFF), (uint8_t)(crc >> 8) };
  io->write(hdr, 5);
  if (len) io->write(payload, len);
  io->write(tail, 2);
}

/* returns cmd (>=0) on a valid frame, -1 on timeout, -2 on bad crc/len */
static int link_recv(const LinkIO* io, uint8_t* payload, uint16_t maxlen,
                     uint16_t* outlen, uint32_t timeout_ms) {
  for (;;) {
    int b = io->read(timeout_ms);
    if (b < 0) return -1;
    if (b != 0xA5) continue;
    b = io->read(timeout_ms);
    if (b < 0) return -1;
    if (b != 0x5A) continue;

    int cmd = io->read(timeout_ms);     if (cmd < 0) return -1;
    int l0  = io->read(timeout_ms);     if (l0 < 0) return -1;
    int l1  = io->read(timeout_ms);     if (l1 < 0) return -1;
    uint16_t len = (uint16_t)(l0 | (l1 << 8));
    if (len > maxlen) return -2;
    for (uint16_t i = 0; i < len; i++) {
      int pb = io->read(timeout_ms);
      if (pb < 0) return -1;
      payload[i] = (uint8_t)pb;
    }
    int c0 = io->read(timeout_ms);      if (c0 < 0) return -1;
    int c1 = io->read(timeout_ms);      if (c1 < 0) return -1;
    uint16_t rx = (uint16_t)(c0 | (c1 << 8));
    if (rx != link_crc((uint8_t)cmd, payload, len)) return -2;
    if (outlen) *outlen = len;
    return cmd;
  }
}
#endif
