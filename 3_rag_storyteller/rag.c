#include "rag.h"
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

#define MAX_PASSAGES 4096
#define MAX_TOKENS 24
#define TOKLEN 24

static RagPassage g_p[MAX_PASSAGES];
static int g_n = 0;
static int g_best_cov = 0;

static const char* STOP[] = {
  "the","a","an","is","are","was","were","do","does","did","to","of","in",
  "on","at","for","and","or","it","its","my","me","i","you","your","can",
  "what","whats","how","why","when","where","who","which","tell","about",
  "much","many","there","that","this","with","from"
};
static int is_stop(const char* w) {
  for (size_t i = 0; i < sizeof(STOP)/sizeof(*STOP); i++)
    if (!strcmp(w, STOP[i])) return 1;
  return 0;
}

int rag_load(const uint8_t* blob, size_t size) {
  if (size < 8) return -1;
  uint32_t magic; memcpy(&magic, blob, 4);
  if (magic != RAG_MAGIC) return -2;
  uint32_t count; memcpy(&count, blob + 4, 4);
  if (count > MAX_PASSAGES) count = MAX_PASSAGES;
  const uint8_t* p = blob + 8;
  const uint8_t* end = blob + size;
  g_n = 0;
  for (uint32_t i = 0; i < count && p + 2 <= end; i++) {
    uint16_t len; memcpy(&len, p, 2); p += 2;
    if (p + len > end) break;
    g_p[g_n].text = (const char*)p;       /* NUL-terminated in blob */
    g_p[g_n].len = len;
    g_n++;
    p += len + 1;
  }
  return g_n;
}

int rag_count(void) { return g_n; }
const RagPassage* rag_get(int idx) {
  return (idx >= 0 && idx < g_n) ? &g_p[idx] : NULL;
}

static int edit_dist(const char* a, const char* b, int maxd) {
  int la = (int)strlen(a), lb = (int)strlen(b);
  if (abs(la - lb) > maxd) return maxd + 1;
  int row[TOKLEN + 1];
  for (int j = 0; j <= lb; j++) row[j] = j;
  for (int i = 1; i <= la; i++) {
    int prev = row[0]; row[0] = i; int rmin = i;
    for (int j = 1; j <= lb; j++) {
      int cur = row[j];
      int cost = (a[i-1] == b[j-1]) ? 0 : 1;
      int v = row[j] + 1;
      if (row[j-1] + 1 < v) v = row[j-1] + 1;
      if (prev + cost < v) v = prev + cost;
      row[j] = v; prev = cur;
      if (v < rmin) rmin = v;
    }
    if (rmin > maxd) return maxd + 1;
  }
  return row[lb];
}

static int words_match(const char* u, const char* k) {
  if (!strcmp(u, k)) return 2;                       /* exact */
  int lu = (int)strlen(u), lk = (int)strlen(k);
  int m = lu < lk ? lu : lk;
  if (lu >= 4 && lk >= 4 && !strncmp(u, k, m)) return 1;   /* prefix/stem */
  if (lu >= 5 && lk >= 5) {
    int maxd = lu >= 8 ? 2 : 1;
    if (edit_dist(u, k, maxd) <= maxd) return 1;     /* typo */
  }
  return 0;
}

static int tokenize(const char* s, char tok[][TOKLEN], int skip_stop) {
  int n = 0, len = 0;
  char buf[TOKLEN];
  for (;; s++) {
    if (*s && (isalnum((unsigned char)*s))) {
      if (len < TOKLEN - 1) buf[len++] = (char)tolower((unsigned char)*s);
    } else {
      if (len > 0) {
        buf[len] = '\0';
        if ((!skip_stop || !is_stop(buf)) && n < MAX_TOKENS)
          strcpy(tok[n++], buf);
        len = 0;
      }
      if (!*s) break;
    }
  }
  return n;
}

/* how many passages contain a query word (fuzzy) — rare words score higher */
static int word_df(const char* w) {
  int df = 0;
  char pt[MAX_TOKENS][TOKLEN];
  for (int i = 0; i < g_n; i++) {
    int np = tokenize(g_p[i].text, pt, 0);
    for (int j = 0; j < np; j++)
      if (words_match(w, pt[j])) { df++; break; }
  }
  return df;
}

int rag_query_cov(const char* query, int* idx, int* score, int k, int* coverage_pct) {
  char qt[MAX_TOKENS][TOKLEN];
  int nq = tokenize(query, qt, 1);
  if (nq == 0 || g_n == 0) return 0;

  /* rarity weights per query word */
  int wgt[MAX_TOKENS];
  for (int t = 0; t < nq; t++) {
    int df = word_df(qt[t]);
    if (df == 0)      wgt[t] = 0;       /* word appears nowhere */
    else if (df == 1) wgt[t] = 16;
    else if (df <= 3) wgt[t] = 10;
    else if (df <= g_n / 4) wgt[t] = 6;
    else              wgt[t] = 2;       /* near-ubiquitous */
  }

  g_best_cov = 0;
  for (int i = 0; i < k; i++) { idx[i] = -1; score[i] = 0; }
  char pt[MAX_TOKENS][TOKLEN];
  for (int i = 0; i < g_n; i++) {
    int np = tokenize(g_p[i].text, pt, 0);
    int s = 0;
    for (int t = 0; t < nq; t++) {
      if (!wgt[t]) continue;
      int best = 0;
      for (int j = 0; j < np; j++) {
        int m = words_match(qt[t], pt[j]);
        if (m > best) best = m;
        if (best == 2) break;
      }
      if (best) s += wgt[t] * best;     /* exact counts double */
    }
    if (s <= 0) continue;
    /* count distinct query words matched, for coverage of the best hit */
    if (coverage_pct) {
      int matched = 0, considered = 0;
      for (int t = 0; t < nq; t++) {
        considered++;   /* count even words absent from the whole corpus */
        for (int j = 0; j < np; j++)
          if (words_match(qt[t], pt[j])) { matched++; break; }
      }
      int cov = considered ? matched * 100 / considered : 0;
      if (idx[0] == -1 || s > score[0]) g_best_cov = cov;
    }
    /* insert into top-k */
    int j = k - 1;
    if (s <= score[j] && idx[j] != -1) continue;
    while (j > 0 && score[j-1] < s) {
      score[j] = score[j-1]; idx[j] = idx[j-1]; j--;
    }
    score[j] = s; idx[j] = i;
  }
  if (coverage_pct) *coverage_pct = g_best_cov;
  int found = 0;
  while (found < k && idx[found] != -1) found++;
  return found;
}

int rag_query(const char* query, int* idx, int* score, int k) {
  return rag_query_cov(query, idx, score, k, NULL);
}
