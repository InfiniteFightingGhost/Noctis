/*
 * ESP32 Flash Data Extractor - Standalone
 * Upload this ONLY to extract data, then re-upload your recording firmware
 */

#include <SPI.h>
#include <SPIMemory.h>

// --- YOUR PIN DEFINITIONS (from document 5) ---
#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   19
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   17

// --- DATA STRUCTURES (Must match your recording firmware) ---
struct __attribute__((packed)) EpochQ15 {
  uint8_t in_bed_pct, hr_mean, hr_std;
  int8_t  dhr;
  uint8_t rr_mean, rr_std;
  int8_t  drr;
  uint8_t large_move_pct, minor_move_pct, turnovers_delta, apnea_delta, flags, vib_move_pct, vib_resp_q, agree_flags;
};

struct __attribute__((packed)) ChunkHeader {
  uint32_t timestamp;
  uint16_t num_epochs;
  uint16_t crc16;
};

struct __attribute__((packed)) DataChunk {
  ChunkHeader header;
  EpochQ15 epochs[20];  // CHUNK_EPOCHS = 20
};

struct FlashMetadata {
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t magic;
};

// --- GLOBALS ---
SPIFlash flash(PIN_FLASH_CS);
FlashMetadata meta;

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n╔════════════════════════════════════╗");
  Serial.println("║   FLASH DATA EXTRACTOR v1.0        ║");
  Serial.println("╚════════════════════════════════════╝");
  
  // Initialize SPI
  SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_FLASH_CS);
  
  // Initialize Flash
  if (!flash.begin()) {
    Serial.println("ERROR: Cannot connect to W25Q64 flash!");
    Serial.println("Check wiring:");
    Serial.println("  CLK  → Pin 18");
    Serial.println("  MISO → Pin 19");
    Serial.println("  MOSI → Pin 23");
    Serial.println("  CS   → Pin 17");
    while (1) delay(1000);
  }
  
  Serial.println("✓ Flash connected");
  Serial.print("  Capacity: ");
  Serial.print(flash.getCapacity());
  Serial.println(" bytes");
  
  // Read metadata
  flash.readAnything(0, meta);
  
  if (meta.magic == 0xDEADBEEF) {
    Serial.println("✓ Valid metadata found");
    Serial.print("  Total chunks: ");
    Serial.println(meta.totalChunks);
    Serial.print("  Write address: ");
    Serial.println(meta.writeAddr);
  } else {
    Serial.println("⚠ No valid metadata - flash may be empty or uninitialized");
    Serial.print("  Magic value: 0x");
    Serial.println(meta.magic, HEX);
  }
  
  Serial.println("\n=== READY FOR EXTRACTION ===");
  Serial.println("Send command:");
  Serial.println("  DUMP     - Download all data");
  Serial.println("  INFO     - Show flash info");
  Serial.println("  SCAN     - Scan for valid chunks");
  Serial.println("============================\n");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "DUMP") {
      dumpAllData();
    } else if (cmd == "INFO") {
      showInfo();
    } else if (cmd == "SCAN") {
      scanChunks();
    } else {
      Serial.println("Unknown command. Use: DUMP, INFO, or SCAN");
    }
  }
}

void dumpAllData() {
  Serial.println("DUMP_START_ACK");
  
  // Determine chunk count
  uint32_t totalChunks = 0;
  if (meta.magic == 0xDEADBEEF) {
    totalChunks = meta.totalChunks;
  } else {
    // Scan for chunks if no metadata
    Serial.println("WARNING: No metadata, scanning...");
    totalChunks = scanChunks();
  }
  
  Serial.print("TOTAL_CHUNKS:");
  Serial.println(totalChunks);
  Serial.print("CHUNK_SIZE:");
  Serial.println(sizeof(DataChunk));
  Serial.println("DATA_BEGIN");
  
  // Stream all chunks
  for (uint32_t i = 0; i < totalChunks; i++) {
    uint32_t addr = 256 + (i * sizeof(DataChunk));
    DataChunk chunk;
    
    // Read chunk from flash
    flash.readByteArray(addr, (uint8_t*)&chunk, sizeof(DataChunk));
    
    // Send chunk marker
    Serial.print("CHUNK:");
    Serial.println(i);
    
    // Send binary data
    Serial.write((uint8_t*)&chunk, sizeof(DataChunk));
    Serial.println();
    
    // Progress indicator
    if ((i + 1) % 10 == 0) {
      delay(10);  // Small delay to prevent buffer overflow
    }
  }
  
  Serial.println("DATA_END");
  Serial.print("TOTAL_BYTES:");
  Serial.println(totalChunks * sizeof(DataChunk));
  Serial.println("DUMP_COMPLETE");
}

void showInfo() {
  Serial.println("\n=== FLASH INFORMATION ===");
  
  Serial.print("Flash Chip: W25Q64\n");
  Serial.print("Capacity: ");
  Serial.print(flash.getCapacity());
  Serial.println(" bytes");
  
  Serial.print("Manufacturer ID: 0x");
  Serial.println(flash.getManID(), HEX);
  
  Serial.print("JEDEC ID: 0x");
  Serial.println(flash.getJEDECID(), HEX);
  
  Serial.println("\n--- Metadata (Address 0x00) ---");
  if (meta.magic == 0xDEADBEEF) {
    Serial.println("Status: VALID");
    Serial.print("Magic: 0x");
    Serial.println(meta.magic, HEX);
    Serial.print("Write Address: ");
    Serial.println(meta.writeAddr);
    Serial.print("Total Chunks: ");
    Serial.println(meta.totalChunks);
    
    float hours = (meta.totalChunks * 20 * 30) / 3600.0;
    Serial.print("Recording Duration: ");
    Serial.print(hours, 1);
    Serial.println(" hours");
  } else {
    Serial.println("Status: INVALID or EMPTY");
    Serial.print("Magic: 0x");
    Serial.println(meta.magic, HEX);
  }
  
  Serial.println("=========================\n");
}

uint32_t scanChunks() {
  Serial.println("\n=== DEEP SCANNING FOR CHUNKS ===");
  Serial.println("Ignoring metadata, scanning entire flash...");
  
  uint32_t chunkCount = 0;
  uint32_t chunkSize = sizeof(DataChunk);
  
  for (uint32_t addr = 256; addr < flash.getCapacity(); addr += chunkSize) {
    ChunkHeader header;
    flash.readByteArray(addr, (uint8_t*)&header, sizeof(ChunkHeader));
    
    // Check if this looks like valid data
    bool validTimestamp = (header.timestamp > 946684800 && header.timestamp < 2147483647);  // Between 2000-2038
    bool validEpochs = (header.num_epochs == 20);
    
    if (validTimestamp && validEpochs) {
      chunkCount++;
      
      if (chunkCount == 1) {
        Serial.print("✓ First valid chunk found at address: ");
        Serial.println(addr);
        Serial.print("  Timestamp: ");
        Serial.println(header.timestamp);
        Serial.print("  Date: ");
        
        // Print readable date
        time_t t = header.timestamp;
        struct tm *timeinfo = localtime(&t);
        char buffer[32];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        Serial.println(buffer);
      }
      
      if (chunkCount % 50 == 0) {
        Serial.print("Found ");
        Serial.print(chunkCount);
        Serial.println(" chunks...");
      }
    } else {
      // Hit empty or invalid data, stop scanning
      if (chunkCount > 0) break;
    }
    
    // Safety check - don't scan forever
    if (addr > 1000000 && chunkCount == 0) {
      Serial.println("Scanned 1MB, no valid chunks found");
      break;
    }
  }
  
  Serial.print("\n✓ SCAN COMPLETE: Found ");
  Serial.print(chunkCount);
  Serial.println(" chunks");
  Serial.println("===========================\n");
  
  // Update metadata with found chunks
  if (chunkCount > 0) {
    meta.totalChunks = chunkCount;
    meta.writeAddr = 256 + (chunkCount * chunkSize);
  }
  
  return chunkCount;
}
