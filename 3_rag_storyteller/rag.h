/*
 * rag.h/rag.c — passage retrieval over a corpus loaded from SD to RAM.
 * Keyword scoring with rarity weighting + fuzzy typo matching.
 * Portable C99 (host-testable), no Arduino dependencies.
 */
#ifndef RAG_H
#define RAG_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define RAG_MAGIC 0x47415253u /* "SRAG" */

typedef struct { const char* text; uint16_t len; } RagPassage;

/* parse corpus blob in place (blob must stay alive). returns count or <0 */
int rag_load(const uint8_t* blob, size_t size);
int rag_count(void);
const RagPassage* rag_get(int idx);

/* score all passages against a query; fills idx[]/score[] with top k
 * (descending). returns number found with score > 0. */
int rag_query(const char* query, int* idx, int* score, int k);
/* same, also reports what fraction (0-100) of meaningful query words the
 * best passage matched — gate answers on this to avoid confident nonsense */
int rag_query_cov(const char* query, int* idx, int* score, int k, int* coverage_pct);

#ifdef __cplusplus
}
#endif
#endif
