#ifndef TYPES_H
#define TYPES_H

#include <Arduino.h>

// --- CONFIGURATION ---
#define CHUNK_EPOCHS          20

// --- DATA STRUCTURES ---
struct __attribute__((packed)) EpochQ15 {
  uint8_t in_bed_pct;
  uint8_t hr_mean;
  uint8_t hr_std;
  int8_t  dhr;
  uint8_t rr_mean;
  uint8_t rr_std;
  int8_t  drr;
  uint8_t large_move_pct;
  uint8_t minor_move_pct;
  uint8_t turnovers_delta;
  uint8_t apnea_delta;
  uint8_t flags;
  uint8_t vib_move_pct;
  uint8_t vib_resp_q;
  uint8_t agree_flags;
};

struct __attribute__((packed)) ChunkHeader {
  uint32_t timestamp;
  uint16_t num_epochs;
  uint16_t crc16;
};

struct __attribute__((packed)) DataChunk {
  ChunkHeader header;
  EpochQ15 epochs[CHUNK_EPOCHS];
};

#define UPLOAD_BITMAP_SIZE 1024 // 1KB bitmap, covers 8192 chunks

struct FlashMetadata {
  uint32_t version;         // = 3 for this version
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t uploadedChunks;
  uint32_t magic;           // 0xDEADBEEF
  char recording_id[37];    // UUID string
  uint8_t upload_bitmap[UPLOAD_BITMAP_SIZE];
};

struct FlashMetadataV2 {
  uint32_t version;         // = 2 for this version
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t uploadedChunks;
  uint32_t magic;           // 0xDEADBEEF
  uint8_t upload_bitmap[UPLOAD_BITMAP_SIZE];
};

struct FlashMetadataV1 {
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t magic;
};

#endif // TYPES_H
