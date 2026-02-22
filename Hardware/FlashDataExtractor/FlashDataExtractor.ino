/*
 * Flash Data Extractor - v1.3
 * Reads metadata from sector 1 (0x1000) to match recording firmware v3.5
 */

#include <SPI.h>
#include <SPIMemory.h>

#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   5
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   19

#define METADATA_ADDR   0x1000   // Must match recording firmware
#define DATA_START_ADDR 0x2000   // Must match recording firmware

struct __attribute__((packed)) EpochQ15 {
  uint8_t in_bed_pct, hr_mean, hr_std;
  int8_t  dhr;
  uint8_t rr_mean, rr_std;
  int8_t  drr;
  uint8_t large_move_pct, minor_move_pct, turnovers_delta,
          apnea_delta, flags, vib_move_pct, vib_resp_q, agree_flags;
};

struct __attribute__((packed)) ChunkHeader {
  uint32_t timestamp;
  uint16_t num_epochs;
  uint16_t crc16;
};

struct __attribute__((packed)) DataChunk {
  ChunkHeader header;
  EpochQ15 epochs[20];
};

struct FlashMetadata {
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t magic;
};

SPIFlash flash(PIN_FLASH_CS);

void setup() {
  Serial.begin(115200);
  delay(2000);

  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║   Flash Data Extractor v1.3        ║");
  Serial.println("╚════════════════════════════════════╝");

  SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_FLASH_CS);
  SPI.setFrequency(1000000);

  if (!flash.begin()) {
    Serial.println("ERROR: Flash not found!");
    while(1) delay(1000);
  }
  Serial.print("✓ Flash OK - "); Serial.print(flash.getCapacity()); Serial.println(" bytes");

  // Read metadata from sector 1
  FlashMetadata meta;
  flash.readAnything(METADATA_ADDR, meta);

  Serial.println("\n--- Metadata (sector 1, addr 0x1000) ---");
  Serial.print("magic:       0x"); Serial.println(meta.magic, HEX);
  Serial.print("writeAddr:   "); Serial.println(meta.writeAddr);
  Serial.print("totalChunks: "); Serial.println(meta.totalChunks);

  if (meta.magic == 0xDEADBEEF) {
    Serial.print("\n✓ Valid - "); Serial.print(meta.totalChunks); Serial.println(" chunk(s) ready");
  } else {
    Serial.println("\n⚠ No valid metadata at 0x1000");
    Serial.println("  Make sure you're using recording firmware v3.5");
  }

  Serial.println("\nCommands: DUMP | INFO | SCAN");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim(); cmd.toUpperCase();

    if      (cmd == "DUMP") dumpAllData();
    else if (cmd == "INFO") showInfo();
    else if (cmd == "SCAN") scanChunks();
    else Serial.println("Commands: DUMP | INFO | SCAN");
  }
}

void dumpAllData() {
  // Always re-read metadata fresh
  FlashMetadata meta;
  flash.readAnything(METADATA_ADDR, meta);

  Serial.println("DUMP_START_ACK");

  uint32_t totalChunks = 0;
  if (meta.magic == 0xDEADBEEF) {
    totalChunks = meta.totalChunks;
  } else {
    Serial.println("WARNING: No metadata, scanning...");
    totalChunks = scanChunks();
  }

  Serial.print("TOTAL_CHUNKS:"); Serial.println(totalChunks);
  Serial.print("CHUNK_SIZE:");   Serial.println(sizeof(DataChunk));
  Serial.println("DATA_BEGIN");

  for (uint32_t i = 0; i < totalChunks; i++) {
    uint32_t addr = DATA_START_ADDR + (i * sizeof(DataChunk));
    DataChunk chunk;
    flash.readByteArray(addr, (uint8_t*)&chunk, sizeof(DataChunk));
    Serial.print("CHUNK:"); Serial.println(i);
    Serial.write((uint8_t*)&chunk, sizeof(DataChunk));
    Serial.println();
    if ((i + 1) % 10 == 0) delay(10);
  }

  Serial.println("DATA_END");
  Serial.print("TOTAL_BYTES:"); Serial.println(totalChunks * sizeof(DataChunk));
  Serial.println("DUMP_COMPLETE");
}

void showInfo() {
  FlashMetadata meta;
  flash.readAnything(METADATA_ADDR, meta);

  Serial.println("\n=== FLASH INFO ===");
  Serial.print("Capacity:  "); Serial.println(flash.getCapacity());
  Serial.print("Manuf ID:  0x"); Serial.println(flash.getManID(), HEX);
  Serial.print("JEDEC ID:  0x"); Serial.println(flash.getJEDECID(), HEX);

  Serial.println("\n--- Metadata raw bytes (0x1000-0x100B) ---");
  for (int i = 0; i < 12; i++) {
    Serial.print("Byte "); Serial.print(i);
    Serial.print(": 0x"); Serial.println(flash.readByte(METADATA_ADDR + i), HEX);
  }

  Serial.println("\n--- Parsed ---");
  Serial.print("magic:       0x"); Serial.println(meta.magic, HEX);
  Serial.print("writeAddr:   "); Serial.println(meta.writeAddr);
  Serial.print("totalChunks: "); Serial.println(meta.totalChunks);
  Serial.print("Valid:       "); Serial.println(meta.magic == 0xDEADBEEF ? "YES" : "NO");

  if (meta.magic == 0xDEADBEEF && meta.totalChunks > 0) {
    Serial.print("Recorded:    ");
    Serial.print(meta.totalChunks * 10.0 / 60.0, 1);
    Serial.println(" hours");

    ChunkHeader h;
    flash.readByteArray(DATA_START_ADDR, (uint8_t*)&h, sizeof(ChunkHeader));
    Serial.print("First chunk ts: "); Serial.println(h.timestamp);
    Serial.print("First chunk crc: 0x"); Serial.println(h.crc16, HEX);
  }
  Serial.println("==================\n");
}

uint32_t scanChunks() {
  Serial.println("\n=== SCANNING from 0x2000 ===");
  uint32_t count = 0;
  uint32_t chunkSize = sizeof(DataChunk);

  for (uint32_t addr = DATA_START_ADDR; addr < flash.getCapacity(); addr += chunkSize) {
    ChunkHeader h;
    flash.readByteArray(addr, (uint8_t*)&h, sizeof(ChunkHeader));

    bool validTs = (h.timestamp > 946684800 && h.timestamp < 2147483647);
    bool validEp = (h.num_epochs == 20);

    if (validTs && validEp) {
      count++;
      if (count == 1) {
        Serial.print("First chunk at: "); Serial.println(addr);
        Serial.print("Timestamp:      "); Serial.println(h.timestamp);
      }
    } else {
      if (count > 0) break;
    }
    if (addr > DATA_START_ADDR + 2000000 && count == 0) {
      Serial.println("Scanned 2MB, nothing found");
      break;
    }
  }

  Serial.print("Found: "); Serial.print(count); Serial.println(" chunks");
  Serial.println("============================\n");
  return count;
}
