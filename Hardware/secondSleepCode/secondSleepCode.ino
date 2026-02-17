/*
 * ESP32 Sleep Monitor - DEBUG VERSION
 * With detailed logging to diagnose save issues
 */

#include <Wire.h>
#include <SPI.h>
#include <RTClib.h>
#include <SPIMemory.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DFRobot_HumanDetection.h>
#include <esp_task_wdt.h>

// --- YOUR PIN DEFINITIONS ---
#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   19
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   17   
#define PIN_I2C_SDA    2    
#define PIN_I2C_SCL    15    
#define PIN_C1001_RX   26  
#define PIN_C1001_TX   27    
#define LED_STATUS_PIN 32     

// --- CONFIGURATION ---
#define MPU_ADDR             0x69
#define METADATA_ADDR        0x00
#define EPOCH_SECONDS        30
#define CHUNK_EPOCHS         20
#define MAX_SAMPLES_PER_EPOCH 35
#define LARGE_MOVE_THRESHOLD 2.5
#define MINOR_MOVE_THRESHOLD 1.2
#define WDT_TIMEOUT_S        30

// --- DATA STRUCTURES ---
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
  EpochQ15 epochs[CHUNK_EPOCHS];
};

struct FlashMetadata {
  uint32_t writeAddr;
  uint32_t totalChunks;
  uint32_t magic; 
};

// --- GLOBALS ---
SPIFlash flash(PIN_FLASH_CS);
RTC_DS3231 rtc;
Adafruit_MPU6050 mpu;
DFRobot_HumanDetection radar(&Serial2);

FlashMetadata meta;
DataChunk currentChunk;
uint8_t hrBuffer[MAX_SAMPLES_PER_EPOCH], rrBuffer[MAX_SAMPLES_PER_EPOCH];
uint8_t bufferIndex = 0, epochIndex = 0, prevHR = 0, prevRR = 0;
uint32_t flashWriteAddr = 256, epochStartTime = 0, vibSamplesTotal = 0;
uint32_t largeMoveCount = 0, minorMoveCount = 0, sumInBed = 0, radarSamples = 0, sumHR = 0, sumRR = 0;

// --- FORWARD DECLARATIONS ---
void printStatus();
void handleTimeSync(String cmd);
void processEpoch();
void saveChunkToFlash();

// --- UTILITIES ---
float calculateStdDev(uint8_t* data, uint8_t count, uint8_t mean) {
  if (count < 2) return 0;
  float sumSquares = 0;
  for (uint8_t i = 0; i < count; i++) {
    float diff = (float)data[i] - mean;
    sumSquares += diff * diff;
  }
  return sqrt(sumSquares / count);
}

uint16_t calculateCRC16(uint8_t *data, uint16_t length) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 0x0001) crc = (crc >> 1) ^ 0xA001;
      else crc >>= 1;
    }
  }
  return crc;
}

void indicateError(uint8_t errorCode) {
  while (1) {
    for (uint8_t i = 0; i < errorCode; i++) {
      digitalWrite(LED_STATUS_PIN, HIGH); delay(150);
      digitalWrite(LED_STATUS_PIN, LOW); delay(150);
    }
    delay(2000); 
    esp_task_wdt_reset();
  }
}

// --- SETUP ---
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, PIN_C1001_RX, PIN_C1001_TX);
  
  pinMode(LED_STATUS_PIN, OUTPUT);
  digitalWrite(LED_STATUS_PIN, LOW);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);
 SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_FLASH_CS);
  SPI.setFrequency(1000000);  // 1 MHz (slower but more reliable)
  SPI.setDataMode(SPI_MODE0);

  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(NULL);

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║  Sleep Monitor DEBUG v3.3        ║");
  Serial.println("╚══════════════════════════════════╝");

  if (!rtc.begin()) { Serial.println("✗ RTC Fail!"); indicateError(1); }
  Serial.println("✓ RTC OK");
  
  if (!mpu.begin(MPU_ADDR, &Wire)) { Serial.println("✗ MPU Fail!"); indicateError(2); }
  Serial.println("✓ MPU6050 OK");
  
if (!flash.begin()) { 
  Serial.println("✗ Flash Fail!"); 
  indicateError(3); 
}

// ADD THIS DIAGNOSTIC:
Serial.println("\n--- Flash Chip Diagnostics ---");
Serial.print("Manufacturer ID: 0x");
Serial.println(flash.getManID(), HEX);
Serial.print("JEDEC ID: 0x");
Serial.println(flash.getJEDECID(), HEX);
Serial.print("Capacity: ");
Serial.println(flash.getCapacity());

// Test write to address 0x1000 (safe test area)
Serial.println("\nTesting write capability...");
uint8_t testData[4] = {0xDE, 0xAD, 0xBE, 0xEF};
uint8_t readBack[4] = {0};

flash.eraseSector(0x1000);
delay(100);

bool testWrite = flash.writeByteArray(0x1000, testData, 4);
Serial.print("Test write: ");
Serial.println(testWrite ? "SUCCESS" : "FAILED");

if (testWrite) {
  flash.readByteArray(0x1000, readBack, 4);
  Serial.print("Read back: 0x");
  for (int i = 0; i < 4; i++) {
    Serial.print(readBack[i], HEX);
  }
  Serial.println();
  
  if (memcmp(testData, readBack, 4) == 0) {
    Serial.println("✓ Flash write/read test PASSED");
  } else {
    Serial.println("✗ Flash verification FAILED");
  }
}
Serial.println("------------------------------\n");

  // Load or initialize metadata
  flash.readAnything(METADATA_ADDR, meta);
  if (meta.magic == 0xDEADBEEF) {
    flashWriteAddr = meta.writeAddr;
    Serial.print("✓ Resuming from chunk: "); Serial.println(meta.totalChunks);
  } else {
    meta.magic = 0xDEADBEEF;
    meta.writeAddr = 256;
    meta.totalChunks = 0;
    flash.writeAnything(METADATA_ADDR, meta);
    Serial.println("✓ Initialized new flash");
  }

  // Initialize sensors
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  radar.begin();
  radar.configWorkMode(radar.eSleepMode);
  delay(500);

  // Success flash
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_STATUS_PIN, HIGH); delay(100);
    digitalWrite(LED_STATUS_PIN, LOW); delay(100);
  }

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║      RECORDING STARTED           ║");
  Serial.println("╚══════════════════════════════════╝");
  Serial.println("Epochs process every 30 seconds");
  Serial.println("Chunks save every 10 minutes (20 epochs)");
  Serial.println("\nCommands:");
  Serial.println("  STATUS - Show system status");
  Serial.println("  DEBUG  - Show debug counters");
  Serial.println("  SET_TIME YYYY-MM-DD HH:MM:SS");
  Serial.println("══════════════════════════════════\n");
  
  epochStartTime = millis();
}

// --- MAIN LOOP (WITH DEBUG) ---
void loop() {
  esp_task_wdt_reset();

  // Serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd == "STATUS") {
      printStatus();
    } else if (cmd.startsWith("SET_TIME ")) {
      handleTimeSync(cmd);
    } else if (cmd == "DEBUG") {
      Serial.println("\n╔══════════════════════════════════╗");
      Serial.println("║        DEBUG INFO                ║");
      Serial.println("╚══════════════════════════════════╝");
      Serial.print("epochIndex: "); Serial.print(epochIndex); 
      Serial.print("/"); Serial.println(CHUNK_EPOCHS);
      Serial.print("Time until next epoch: ");
      Serial.print((EPOCH_SECONDS * 1000 - (millis() - epochStartTime)) / 1000);
      Serial.println(" seconds");
      Serial.print("bufferIndex: "); Serial.println(bufferIndex);
      Serial.print("radarSamples: "); Serial.println(radarSamples);
      Serial.print("vibSamplesTotal: "); Serial.println(vibSamplesTotal);
      Serial.print("flashWriteAddr: "); Serial.println(flashWriteAddr);
      Serial.println("══════════════════════════════════\n");
    }
  }

  // MPU Sampling @ 10Hz
  static uint32_t lastMPU = 0;
  if (millis() - lastMPU >= 100) {
    lastMPU = millis();
    
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    
    float mag = sqrt(sq(a.acceleration.x) + sq(a.acceleration.y) + sq(a.acceleration.z));
    float move = abs(mag - 9.81);
    
    vibSamplesTotal++;
    if (move > LARGE_MOVE_THRESHOLD) largeMoveCount++;
    else if (move > MINOR_MOVE_THRESHOLD) minorMoveCount++;
  }

  // Radar Sampling @ 1Hz
  static uint32_t lastRadar = 0;
  if (millis() - lastRadar >= 1000) {
    lastRadar = millis();
    
    int hr = radar.getHeartRate();
    int rr = radar.getBreatheValue();
    
    if (hr > 30 && hr < 200 && bufferIndex < MAX_SAMPLES_PER_EPOCH) {
      hrBuffer[bufferIndex] = hr;
      rrBuffer[bufferIndex] = rr;
      sumHR += hr;
      sumRR += rr;
      bufferIndex++;
    }
    
    if (radar.smHumanData(radar.eHumanPresence)) sumInBed++;
    radarSamples++;
  }

  // Process Epoch (every 30 seconds) - WITH DEBUG
  if (millis() - epochStartTime >= (EPOCH_SECONDS * 1000)) {
    Serial.print("\n>>> Epoch "); 
    Serial.print(epochIndex + 1);
    Serial.print("/");
    Serial.print(CHUNK_EPOCHS);
    Serial.print(" (");
    Serial.print((millis() - epochStartTime) / 1000);
    Serial.println("s)");
    
    processEpoch();
    epochStartTime = millis();
  }
}

// --- EPOCH PROCESSING (WITH DEBUG) ---
void processEpoch() {
  EpochQ15 ep;
  
  ep.in_bed_pct = (sumInBed * 100) / (radarSamples > 0 ? radarSamples : 1);
  ep.hr_mean = bufferIndex > 0 ? sumHR / bufferIndex : 0;
  ep.rr_mean = bufferIndex > 0 ? sumRR / bufferIndex : 0;
  
  Serial.print("    HR="); Serial.print(ep.hr_mean);
  Serial.print(" RR="); Serial.print(ep.rr_mean);
  Serial.print(" InBed="); Serial.print(ep.in_bed_pct);
  Serial.print("% Vib="); Serial.print((vibSamplesTotal > 0 ? (largeMoveCount + minorMoveCount) * 100 / vibSamplesTotal : 0));
  Serial.println("%");
  
  ep.hr_std = (uint8_t)calculateStdDev(hrBuffer, bufferIndex, ep.hr_mean);
  ep.rr_std = (uint8_t)calculateStdDev(rrBuffer, bufferIndex, ep.rr_mean);
  
  ep.dhr = (int8_t)constrain((int)ep.hr_mean - prevHR, -128, 127);
  ep.drr = (int8_t)constrain((int)ep.rr_mean - prevRR, -128, 127);
  
  uint32_t totalVib = (vibSamplesTotal > 0 ? vibSamplesTotal : 1);
  ep.large_move_pct = (largeMoveCount * 100) / totalVib;
  ep.minor_move_pct = (minorMoveCount * 100) / totalVib;
  ep.vib_move_pct = ep.large_move_pct + ep.minor_move_pct;
  
  ep.turnovers_delta = 0;
  ep.apnea_delta = radar.smSleepData(radar.eAbnormalStruggle);
  ep.flags = radar.smSleepData(radar.eSleepState);
  ep.vib_resp_q = 50;

  ep.agree_flags = 0;
  if (ep.flags > 0 && ep.vib_move_pct > 20) ep.agree_flags |= 0x01;
  if (ep.in_bed_pct < 50 && ep.vib_move_pct > 10) ep.agree_flags |= 0x02;
  if (radarSamples < 20) ep.agree_flags |= 0x04;

  currentChunk.epochs[epochIndex++] = ep;
  
  prevHR = ep.hr_mean;
  prevRR = ep.rr_mean;
  
  bufferIndex = 0; sumHR = 0; sumRR = 0; sumInBed = 0;
  radarSamples = 0; vibSamplesTotal = 0; largeMoveCount = 0; minorMoveCount = 0;

  if (epochIndex >= CHUNK_EPOCHS) {
    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  CHUNK FULL - SAVING TO FLASH    ║");
    Serial.println("╚══════════════════════════════════╝");
    saveChunkToFlash();
    epochIndex = 0;
  } else {
    Serial.print("    Progress: ");
    Serial.print(epochIndex);
    Serial.print("/");
    Serial.println(CHUNK_EPOCHS);
  }
}

void saveChunkToFlash() {
  currentChunk.header.timestamp = rtc.now().unixtime() - (EPOCH_SECONDS * CHUNK_EPOCHS);
  currentChunk.header.num_epochs = CHUNK_EPOCHS;
  currentChunk.header.crc16 = calculateCRC16((uint8_t*)&currentChunk.epochs, sizeof(currentChunk.epochs));
  
  Serial.print("Write address: "); Serial.println(flashWriteAddr);
  
  // Check if this address is empty (all 0xFF)
  uint8_t testByte = flash.readByte(flashWriteAddr);
  
  if (testByte != 0xFF) {
    Serial.println("⚠ WARNING: Writing to non-empty memory!");
    Serial.println("  This will corrupt data. Erasing sector...");
    
    uint32_t sectorAddr = (flashWriteAddr / 4096) * 4096;
    flash.eraseSector(sectorAddr);
    delay(100);
    Serial.println("✓ Sector erased");
  }
  
  uint8_t* dataPtr = (uint8_t*)&currentChunk;
  uint16_t dataSize = sizeof(DataChunk);
  
  bool writeSuccess = flash.writeByteArray(flashWriteAddr, dataPtr, dataSize);
  
  if (writeSuccess) {
    Serial.println("✓ Flash write OK");
    
    // Verify
    uint8_t verifyBuffer[sizeof(DataChunk)];
    flash.readByteArray(flashWriteAddr, verifyBuffer, dataSize);
    DataChunk* verify = (DataChunk*)verifyBuffer;
    
    if (verify->header.timestamp == currentChunk.header.timestamp) {
      Serial.println("✓ Verification OK");
      
      flashWriteAddr += sizeof(DataChunk);
      meta.writeAddr = flashWriteAddr;
      meta.totalChunks++;
      
      uint8_t* metaPtr = (uint8_t*)&meta;
      flash.writeByteArray(METADATA_ADDR, metaPtr, sizeof(FlashMetadata));
      
      Serial.print("✓✓✓ CHUNK #"); Serial.print(meta.totalChunks);
      Serial.print(" SAVED ("); Serial.print(meta.totalChunks * 10.0 / 60.0, 1);
      Serial.println(" hours) ✓✓✓");
    }
  } else {
    Serial.println("✗✗✗ FLASH WRITE FAILED ✗✗✗");
  }
  
  Serial.println("══════════════════════════════════\n");
}

void printStatus() {
  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║       SYSTEM STATUS              ║");
  Serial.println("╚══════════════════════════════════╝");
  DateTime now = rtc.now();
  Serial.print("Time: "); Serial.println(now.timestamp());
  Serial.print("Flash: "); Serial.print(flashWriteAddr); 
  Serial.print(" / "); Serial.println(flash.getCapacity());
  Serial.print("Chunks saved: "); Serial.println(meta.totalChunks);
  Serial.print("Current epoch: "); Serial.print(epochIndex); 
  Serial.print("/"); Serial.println(CHUNK_EPOCHS);
  Serial.print("Recording time: "); 
  Serial.print(meta.totalChunks * 10.0 / 60.0, 1); 
  Serial.println(" hours");
  Serial.print("Radar presence: "); 
  Serial.println(radar.smHumanData(radar.eHumanPresence) ? "YES" : "NO");
  Serial.print("Radar HR: "); 
  Serial.print(radar.getHeartRate()); 
  Serial.println(" BPM");
  Serial.println("══════════════════════════════════\n");
}

void handleTimeSync(String cmd) {
  int y, m, d, hh, mm, ss;
  if (sscanf(cmd.c_str(), "SET_TIME %d-%d-%d %d:%d:%d", &y, &m, &d, &hh, &mm, &ss) == 6) {
    rtc.adjust(DateTime(y, m, d, hh, mm, ss));
    Serial.println("✓ Time synced");
    Serial.print("  New time: "); Serial.println(rtc.now().timestamp());
  } else {
    Serial.println("✗ Invalid format");
    Serial.println("  Use: SET_TIME YYYY-MM-DD HH:MM:SS");
  }
}