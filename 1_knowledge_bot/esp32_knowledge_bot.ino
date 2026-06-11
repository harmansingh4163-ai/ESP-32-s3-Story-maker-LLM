/*
 * ESP32-S3 Offline Knowledge Bot
 * --------------------------------
 * A fully offline, retrieval-based chat bot for ESP32-S3.
 * No WiFi, no cloud, no LLM weights — a curated knowledge base
 * with keyword scoring + fuzzy matching, so it tolerates typos
 * and loose phrasing while staying instant and deterministic.
 *
 * Chat over Serial Monitor @ 115200 baud (newline-terminated).
 * Hook botRespond() into LVGL/touchscreen UI later if desired.
 *
 * Board: any ESP32 / ESP32-S3 (tested target: ESP32-S3, Arduino core)
 */

#include <Arduino.h>
#include <ctype.h>
#include <string.h>

// ============================================================
// KNOWLEDGE BASE
// Add entries here. keywords = space-separated lowercase words
// that should trigger this answer. More specific keywords rank
// higher automatically (rarity weighting).
// ============================================================

struct KBEntry {
  const char* keywords;   // trigger words
  const char* question;   // canonical question (shown as suggestion)
  const char* answer;
};

static const KBEntry KB[] = {
  // --- Sample domain: general science/tech facts. Replace with your own. ---
  {"esp32 chip what processor cores speed",
   "What is the ESP32?",
   "The ESP32 is a low-cost microcontroller by Espressif with dual Xtensa cores (up to 240 MHz), WiFi, and Bluetooth. The S3 variant adds vector instructions and better PSRAM support."},

  {"speed light fast travel",
   "How fast is the speed of light?",
   "Light travels at 299,792 km per second in a vacuum. That's about 7.5 trips around Earth in one second."},

  {"water boil temperature boiling point",
   "At what temperature does water boil?",
   "Water boils at 100\xC2\xB0" "C (212\xC2\xB0" "F) at sea level. At higher altitudes it boils at lower temperatures because air pressure drops."},

  {"moon distance earth far",
   "How far is the Moon from Earth?",
   "The Moon is about 384,400 km from Earth on average. Light takes roughly 1.3 seconds to cover that distance."},

  {"photosynthesis plants make food sunlight",
   "How do plants make food?",
   "Plants use photosynthesis: they capture sunlight with chlorophyll and convert water and CO2 into glucose and oxygen."},

  {"gravity why things fall newton",
   "What is gravity?",
   "Gravity is the attraction between masses. Earth's gravity accelerates falling objects at about 9.8 m/s squared."},

  {"binary computer count zeros ones bits",
   "How do computers count?",
   "Computers use binary: everything is represented with 0s and 1s. Each binary digit is a bit; 8 bits make a byte."},

  {"heart beats per minute pulse human",
   "How fast does the human heart beat?",
   "A resting adult heart beats about 60-100 times per minute, pumping roughly 5 liters of blood every minute."},

  {"tallest mountain everest height",
   "What is the tallest mountain?",
   "Mount Everest is the tallest mountain above sea level at 8,849 meters. Measured from base to peak, Mauna Kea in Hawaii is taller."},

  {"deepest ocean trench mariana",
   "What is the deepest part of the ocean?",
   "The Mariana Trench's Challenger Deep is about 10,935 meters down. Pressure there is over 1,000 times sea-level pressure."},

  {"wifi how works wireless radio",
   "How does WiFi work?",
   "WiFi sends data using radio waves, typically at 2.4 GHz or 5 GHz. Devices encode data onto these waves and a router relays it to the network."},

  {"battery how works store energy lithium",
   "How do batteries work?",
   "Batteries store chemical energy and convert it to electricity. Ions move between two electrodes through an electrolyte, pushing electrons through your circuit."},

  {"dinosaurs extinct asteroid when died",
   "Why did the dinosaurs go extinct?",
   "About 66 million years ago an asteroid roughly 10 km wide struck near today's Yucatan Peninsula, causing climate collapse that wiped out most dinosaurs."},

  {"sun star hot temperature core",
   "How hot is the Sun?",
   "The Sun's surface is about 5,500\xC2\xB0" "C, while its core reaches around 15 million \xC2\xB0" "C, where hydrogen fuses into helium."},

  {"rainbow colors why form light",
   "Why do rainbows form?",
   "Rainbows form when sunlight enters water droplets, bends, reflects internally, and splits into its component colors."},

  {"who you are bot name what",
   "Who are you?",
   "I'm an offline knowledge bot running entirely on this ESP32-S3. No internet, no cloud — just a knowledge base in flash and some clever matching."},

  {"help can do what topics know",
   "What can you do?",
   "Ask me about any topic in my knowledge base — science, tech, nature facts. I match your question to what I know. Type 'topics' to list everything."},
};

static const int KB_SIZE = sizeof(KB) / sizeof(KB[0]);

// ============================================================
// SMALL TALK / FALLBACKS
// ============================================================

struct Pattern { const char* trigger; const char* reply; };

static const Pattern SMALLTALK[] = {
  {"hello hi hey howdy",        "Hey! Ask me anything, or type 'topics' to see what I know."},
  {"thanks thank appreciated",  "You're welcome!"},
  {"bye goodbye exit quit",     "See you! I'll be right here in flash memory."},
  {"how are you doing",         "Running cool at 240 MHz. What would you like to know?"},
};
static const int SMALLTALK_SIZE = sizeof(SMALLTALK) / sizeof(SMALLTALK[0]);

static const char* FALLBACKS[] = {
  "I don't have that in my knowledge base yet.",
  "Hmm, that's outside what I know.",
  "Not sure about that one.",
};

// ============================================================
// TEXT UTILITIES
// ============================================================

#define MAX_INPUT 256
#define MAX_TOKENS 24
#define MAX_TOKEN_LEN 24

// very common words that carry no meaning for matching
static const char* STOPWORDS[] = {
  "the","a","an","is","are","was","were","do","does","did","to","of",
  "in","on","at","for","and","or","it","its","my","me","i","you","your",
  "can","could","would","should","tell","about","please","what","whats",
  "how","why","when","where","who","which","does","much","many","there"
};
static const int STOPWORDS_SIZE = sizeof(STOPWORDS) / sizeof(STOPWORDS[0]);

static bool isStopword(const char* w) {
  for (int i = 0; i < STOPWORDS_SIZE; i++)
    if (strcmp(w, STOPWORDS[i]) == 0) return true;
  return false;
}

// lowercase + strip punctuation in place
static void normalize(char* s) {
  char* dst = s;
  for (char* p = s; *p; p++) {
    if (isalnum((unsigned char)*p)) *dst++ = tolower((unsigned char)*p);
    else if (*p == ' ' || *p == '\t') {
      if (dst != s && *(dst - 1) != ' ') *dst++ = ' ';
    }
  }
  *dst = '\0';
}

// split into tokens, dropping stopwords. Returns token count.
static int tokenize(const char* s, char tokens[][MAX_TOKEN_LEN]) {
  int count = 0, len = 0;
  char buf[MAX_TOKEN_LEN];
  for (const char* p = s; ; p++) {
    if (*p && *p != ' ') {
      if (len < MAX_TOKEN_LEN - 1) buf[len++] = *p;
    } else {
      if (len > 0) {
        buf[len] = '\0';
        if (!isStopword(buf) && count < MAX_TOKENS) {
          strcpy(tokens[count++], buf);
        }
        len = 0;
      }
      if (!*p) break;
    }
  }
  return count;
}

// Levenshtein edit distance with early exit; for typo tolerance.
static int editDistance(const char* a, const char* b, int maxDist) {
  int la = strlen(a), lb = strlen(b);
  if (abs(la - lb) > maxDist) return maxDist + 1;
  static int row[MAX_TOKEN_LEN + 1];
  for (int j = 0; j <= lb; j++) row[j] = j;
  for (int i = 1; i <= la; i++) {
    int prev = row[0];
    row[0] = i;
    int rowMin = row[0];
    for (int j = 1; j <= lb; j++) {
      int cur = row[j];
      int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
      row[j] = min(min(row[j] + 1, row[j - 1] + 1), prev + cost);
      prev = cur;
      if (row[j] < rowMin) rowMin = row[j];
    }
    if (rowMin > maxDist) return maxDist + 1; // early exit
  }
  return row[lb];
}

// fuzzy word match: exact, prefix (stemming-lite), or small edit distance
static bool wordsMatch(const char* userWord, const char* kbWord) {
  if (strcmp(userWord, kbWord) == 0) return true;
  int lu = strlen(userWord), lk = strlen(kbWord);
  // prefix match handles plurals/verb forms: "boiling" ~ "boil"
  if (lu >= 4 && lk >= 4) {
    if (strncmp(userWord, kbWord, min(lu, lk)) == 0) return true;
  }
  // typo tolerance scaled by word length
  if (lu >= 5 && lk >= 5) {
    int maxDist = (lu >= 8) ? 2 : 1;
    if (editDistance(userWord, kbWord, maxDist) <= maxDist) return true;
  }
  return false;
}

// ============================================================
// RETRIEVAL ENGINE
// ============================================================

// Count how many KB entries contain a given word — rare words score higher.
static int keywordRarity(const char* word) {
  int docs = 0;
  char kw[MAX_TOKEN_LEN];
  for (int i = 0; i < KB_SIZE; i++) {
    const char* p = KB[i].keywords;
    bool found = false;
    while (*p && !found) {
      int len = 0;
      while (*p && *p != ' ') { if (len < MAX_TOKEN_LEN - 1) kw[len++] = *p; p++; }
      kw[len] = '\0';
      if (len > 0 && strcmp(kw, word) == 0) found = true;
      while (*p == ' ') p++;
    }
    if (found) docs++;
  }
  return docs;
}

// Score one KB entry against user tokens.
static int scoreEntry(const KBEntry& e, char tokens[][MAX_TOKEN_LEN], int nTokens) {
  int score = 0;
  char kw[MAX_TOKEN_LEN];
  for (int t = 0; t < nTokens; t++) {
    const char* p = e.keywords;
    while (*p) {
      int len = 0;
      while (*p && *p != ' ') { if (len < MAX_TOKEN_LEN - 1) kw[len++] = *p; p++; }
      kw[len] = '\0';
      while (*p == ' ') p++;
      if (len == 0) continue;
      if (wordsMatch(tokens[t], kw)) {
        // exact match worth more than fuzzy; rare words worth more than common
        int base = (strcmp(tokens[t], kw) == 0) ? 10 : 6;
        int rarity = keywordRarity(kw);
        if (rarity <= 1) base += 6;       // unique to this entry
        else if (rarity <= 3) base += 3;  // fairly distinctive
        score += base;
        break; // each user token counts once per entry
      }
    }
  }
  return score;
}

// Find best entry; returns index or -1. Outputs best score + runner-up.
static int retrieve(char tokens[][MAX_TOKEN_LEN], int nTokens, int& bestScore, int& secondIdx) {
  int bestIdx = -1; bestScore = 0; secondIdx = -1; int secondScore = 0;
  for (int i = 0; i < KB_SIZE; i++) {
    int s = scoreEntry(KB[i], tokens, nTokens);
    if (s > bestScore) {
      secondScore = bestScore; secondIdx = bestIdx;
      bestScore = s; bestIdx = i;
    } else if (s > secondScore) {
      secondScore = s; secondIdx = i;
    }
  }
  if (secondScore < bestScore / 2) secondIdx = -1; // runner-up only if close
  return bestIdx;
}

// ============================================================
// BOT LOGIC
// ============================================================

static int lastSuggestion = -1; // for "yes" follow-up

static int countWords(const char* s) {
  int n = 0; bool in = false;
  for (; *s; s++) {
    if (*s != ' ' && !in) { n++; in = true; }
    else if (*s == ' ') in = false;
  }
  return n;
}

static bool matchPattern(const char* trigger, char tokens[][MAX_TOKEN_LEN], int nTokens,
                         const char* rawNormalized) {
  // Small talk matches on raw words (stopwords matter: "hi", "how are you").
  // To avoid hijacking real questions ("HOW fast is light"), require either
  // 2+ trigger-word hits, or a hit on a very short input (<= 2 words).
  char tw[MAX_TOKEN_LEN];
  const char* p = trigger;
  int hits = 0;
  while (*p) {
    int len = 0;
    while (*p && *p != ' ') { if (len < MAX_TOKEN_LEN - 1) tw[len++] = *p; p++; }
    tw[len] = '\0';
    while (*p == ' ') p++;
    // word-boundary search in normalized input
    const char* hay = rawNormalized;
    int twLen = strlen(tw);
    while ((hay = strstr(hay, tw)) != nullptr) {
      bool startOk = (hay == rawNormalized) || (*(hay - 1) == ' ');
      bool endOk = (hay[twLen] == '\0') || (hay[twLen] == ' ');
      if (startOk && endOk) { hits++; break; }
      hay += twLen;
    }
  }
  int inputWords = countWords(rawNormalized);
  return (hits >= 2) || (hits >= 1 && inputWords <= 2);
}

static void listTopics() {
  Serial.println("\nBot: Here's everything I know about:");
  for (int i = 0; i < KB_SIZE; i++) {
    Serial.print("  ");
    Serial.print(i + 1);
    Serial.print(". ");
    Serial.println(KB[i].question);
  }
  Serial.println();
}

// Main entry point — also callable from a touchscreen UI.
// Writes the reply to Serial; returns the answer string (or nullptr).
const char* botRespond(const char* rawInput) {
  static char input[MAX_INPUT];
  strncpy(input, rawInput, MAX_INPUT - 1);
  input[MAX_INPUT - 1] = '\0';
  normalize(input);

  if (strlen(input) == 0) return nullptr;

  // commands
  if (strcmp(input, "topics") == 0 || strcmp(input, "list") == 0) {
    listTopics();
    return nullptr;
  }

  // "yes" follow-up to a suggestion
  if (lastSuggestion >= 0 &&
      (strcmp(input, "yes") == 0 || strcmp(input, "yeah") == 0 ||
       strcmp(input, "sure") == 0 || strcmp(input, "ok") == 0)) {
    const char* ans = KB[lastSuggestion].answer;
    Serial.print("\nBot: "); Serial.println(ans); Serial.println();
    lastSuggestion = -1;
    return ans;
  }

  char tokens[MAX_TOKENS][MAX_TOKEN_LEN];
  int nTokens = tokenize(input, tokens);

  // small talk first
  for (int i = 0; i < SMALLTALK_SIZE; i++) {
    if (matchPattern(SMALLTALK[i].trigger, tokens, nTokens, input)) {
      Serial.print("\nBot: "); Serial.println(SMALLTALK[i].reply); Serial.println();
      lastSuggestion = -1;
      return SMALLTALK[i].reply;
    }
  }

  // knowledge retrieval
  int bestScore, secondIdx;
  int bestIdx = retrieve(tokens, nTokens, bestScore, secondIdx);

  // confidence threshold scales with how much the user actually said
  int threshold = (nTokens <= 1) ? 10 : 13;

  if (bestIdx >= 0 && bestScore >= threshold) {
    Serial.print("\nBot: ");
    Serial.println(KB[bestIdx].answer);
    if (secondIdx >= 0 && secondIdx != bestIdx) {
      Serial.print("     (Related: \"");
      Serial.print(KB[secondIdx].question);
      Serial.println("\")");
    }
    Serial.println();
    lastSuggestion = -1;
    return KB[bestIdx].answer;
  }

  // weak match -> suggest instead of answering wrong
  if (bestIdx >= 0 && bestScore > 0) {
    Serial.print("\nBot: ");
    Serial.print(FALLBACKS[millis() % 3]);
    Serial.print(" Did you mean: \"");
    Serial.print(KB[bestIdx].question);
    Serial.println("\"? (yes/no)\n");
    lastSuggestion = bestIdx;
    return nullptr;
  }

  // nothing at all
  Serial.print("\nBot: ");
  Serial.print(FALLBACKS[millis() % 3]);
  Serial.println(" Type 'topics' to see what I can answer.\n");
  lastSuggestion = -1;
  return nullptr;
}

// ============================================================
// ARDUINO ENTRY POINTS
// ============================================================

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=========================================");
  Serial.println("  ESP32-S3 Offline Knowledge Bot");
  Serial.println("  100% local. No WiFi. No cloud.");
  Serial.println("=========================================");
  Serial.printf("  Knowledge base: %d entries, %u bytes free heap\n",
                KB_SIZE, ESP.getFreeHeap());
  Serial.println("  Ask a question, or type 'topics'.\n");
}

void loop() {
  static char line[MAX_INPUT];
  static int pos = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (pos > 0) {
        line[pos] = '\0';
        Serial.print("You: ");
        Serial.println(line);
        botRespond(line);
        pos = 0;
      }
    } else if (pos < MAX_INPUT - 1) {
      line[pos++] = c;
    }
  }
}
