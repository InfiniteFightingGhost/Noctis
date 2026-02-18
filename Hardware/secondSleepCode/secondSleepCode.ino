/*
 * ESP32 Sleep Monitor - v3.5
 *
 * ROOT CAUSE FIX: Metadata was being written twice to the same flash
 * address without erasing between writes. Flash bits can only change
 * 1→0, never 0→1. So totalChunks=0 AND totalChunks=1 = 0. Extractor
 * always saw 0 chunks even though chunk data was saved correctly.
 *
 * FIX: Metadata lives in its own dedicated sector (sector 1, addr 0x1000).
 * That sector is ALWAYS erased before every metadata write. Chunk data
 * starts at sector 2 (addr 0x2000) and is completely untouched by this.
 *
 * Flash layout:
 *   Sector 0  (0x0000-0x0FFF): unused / reserved
 *   Sector 1  (0x1000-0x1FFF): metadata ONLY - erased on every update
 *   Sector 2+ (0x2000+):       chunk data, never erased unless full
 */

#include <Wire.h>
#include <SPI.h>
#include <RTClib.h>
#include <SPIMemory.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DFRobot_HumanDetection.h>
#include <esp_task_wdt.h>

// --- PIN DEFINITIONS ---
#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   19
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   17
#define PIN_I2C_SDA    2
#define PIN_I2C_SCL    15
#define PIN_C1001_RX   26
#define PIN_C1001_TX   27
#define LED_STATUS_PIN 32

// --- FLASH LAYOUT ---
#define METADATA_ADDR   0x1000   // Sector 1 - metadata only
#define DATA_START_ADDR 0x2000   // Sector 2 - all chunk data starts here

// --- CONFIGURATION ---
#define MPU_ADDR              0x69
#define EPOCH_SECONDS         30
#define CHUNK_EPOCHS          20
#define MAX_SAMPLES_PER_EPOCH 35
#define WDT_TIMEOUT_S         30

// --- VIBRATION THRESHOLDS ---
// Watch "VIB rms=" in Serial Monitor and tune:
//   Still        → rms should be < VIB_MINOR_THRESHOLD
//   Sleep move   → rms between thresholds
//   Violent shake → rms should be >> VIB_LARGE_THRESHOLD
#define VIB_LARGE_THRESHOLD 0.8
#define VIB_MINOR_THRESHOLD 0.15

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

struct FlashMetadata {
  uint32_t writeAddr;    // next address to write chunk data
  uint32_t totalChunks;  // total chunks saved
  uint32_t magic;        // 0xDEADBEEF = valid
};

// --- GLOBALS ---
SPIFlash flash(PIN_FLASH_CS);
RTC_DS3231 rtc;
Adafruit_MPU6050 mpu;
DFRobot_HumanDetection radar(&Serial2);

FlashMetadata meta;
DataChunk currentChunk;

uint8_t hrBuffer[MAX_SAMPLES_PER_EPOCH];
uint8_t rrBuffer[MAX_SAMPLES_PER_EPOCH];
uint8_t bufferIndex = 0;
uint8_t prevHR = 0, prevRR = 0;
uint8_t epochIndex = 0;
uint32_t epochStartTime = 0;
uint32_t flashWriteAddr = DATA_START_ADDR;

uint32_t sumInBed = 0, radarSamples = 0;
uint32_t sumHR = 0, sumRR = 0;
uint8_t  lastTurnovers = 0, lastApnea = 0;

float    lastAccelX = 0, lastAccelY = 0, lastAccelZ = 0;
#define  RMS_WINDOW 10
float    rmsWindow[RMS_WINDOW] = {0};
uint8_t  rmsIndex = 0;
float    vibScoreSum = 0;
uint32_t vibLargeCount = 0, vibMinorCount = 0, vibTotalSamples = 0;

uint32_t lastErasedDataSector = 0xFFFFFFFF;

// --- FORWARD DECLARATIONS ---
void     saveMetadata();
void     printStatus();
void     handleTimeSync(String cmd);
void     processEpoch();
void     saveChunkToFlash();
void     indicateError(uint8_t errorCode);
float    calculateStdDev(uint8_t* data, uint8_t count, uint8_t mean);
uint16_t calculateCRC16(uint8_t* data, uint16_t length);

// --- UTILITIES ---
float calculateStdDev(uint8_t* data, uint8_t count, uint8_t mean) {
  if (count < 2) return 0;
  float sumSq = 0;
  for (uint8_t i = 0; i < count; i++) {
    float d = (float)data[i] - mean;
    sumSq += d * d;
  }
  return sqrt(sumSq / count);
}

uint16_t calculateCRC16(uint8_t* data, uint16_t length) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < length; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 1) crc = (crc >> 1) ^ 0xA001;
      else         crc >>= 1;
    }
  }
  return crc;
}

void indicateError(uint8_t code) {
  while (1) {
    for (uint8_t i = 0; i < code; i++) {
      digitalWrite(LED_STATUS_PIN, HIGH); delay(150);
      digitalWrite(LED_STATUS_PIN, LOW);  delay(150);
    }
    delay(2000);
    esp_task_wdt_reset();
  }
}

// THE KEY FIX: always erase sector 1 before writing metadata.
// Sector 1 contains ONLY metadata so erasing it never touches chunk data.
void saveMetadata() {
  flash.eraseSector(METADATA_ADDR);  // erase sector 1 (0x1000-0x1FFF)
  delay(50);
  flash.writeByteArray(METADATA_ADDR, (uint8_t*)&meta, sizeof(FlashMetadata));
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
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE0);

  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(NULL);

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║   Sleep Monitor v3.5             ║");
  Serial.println("╚══════════════════════════════════╝");

  if (!rtc.begin())             { Serial.println("✗ RTC fail!");    indicateError(1); }
  Serial.println("✓ RTC OK");

  if (!mpu.begin(MPU_ADDR, &Wire)) { Serial.println("✗ MPU fail!"); indicateError(2); }
  Serial.println("✓ MPU6050 OK");
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  if (!flash.begin())           { Serial.println("✗ Flash fail!"); indicateError(3); }
  Serial.print("✓ Flash OK - ");
  Serial.print(flash.getCapacity()); Serial.println(" bytes");

  // Read metadata from sector 1
  flash.readAnything(METADATA_ADDR, meta);

  if (meta.magic == 0xDEADBEEF) {
    flashWriteAddr = meta.writeAddr;
    Serial.print("✓ Resuming - chunks saved: "); Serial.println(meta.totalChunks);
    Serial.print("  Next write addr:          "); Serial.println(flashWriteAddr);
  } else {
    Serial.println("✓ Fresh start - initializing");
    meta.magic       = 0xDEADBEEF;
    meta.writeAddr   = DATA_START_ADDR;
    meta.totalChunks = 0;
    flashWriteAddr   = DATA_START_ADDR;
    saveMetadata();  // safe: erases sector 1 then writes fresh
    Serial.println("✓ Metadata written to sector 1 (0x1000)");
  }

  radar.begin();
  radar.configWorkMode(radar.eSleepMode);
  delay(500);
  Serial.println("✓ Radar OK");

  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_STATUS_PIN, HIGH); delay(100);
    digitalWrite(LED_STATUS_PIN, LOW);  delay(100);
  }

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║      RECORDING STARTED           ║");
  Serial.println("╚══════════════════════════════════╝");
  Serial.println("Epoch every 30s | Chunk every 10min");
  Serial.println("Commands: STATUS | DEBUG | ERASE | SET_TIME YYYY-MM-DD HH:MM:SS");
  Serial.println("══════════════════════════════════\n");

  epochStartTime = millis();
}

// --- MAIN LOOP ---
void loop() {
  esp_task_wdt_reset();

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "STATUS") {
      printStatus();
    } else if (cmd.startsWith("SET_TIME ")) {
      handleTimeSync(cmd);
    } else if (cmd == "DEBUG") {
      Serial.println("\n--- DEBUG ---");
      Serial.print("epochIndex:     "); Serial.print(epochIndex); Serial.print("/"); Serial.println(CHUNK_EPOCHS);
      Serial.print("Next epoch in:  "); Serial.print((EPOCH_SECONDS*1000-(millis()-epochStartTime))/1000); Serial.println("s");
      Serial.print("bufferIndex:    "); Serial.println(bufferIndex);
      Serial.print("radarSamples:   "); Serial.println(radarSamples);
      Serial.print("vibTotal:       "); Serial.println(vibTotalSamples);
      Serial.print("vibLarge:       "); Serial.println(vibLargeCount);
      Serial.print("vibMinor:       "); Serial.println(vibMinorCount);
      Serial.print("flashWriteAddr: "); Serial.println(flashWriteAddr);
      Serial.print("totalChunks:    "); Serial.println(meta.totalChunks);
      Serial.println("-------------\n");
    } else if (cmd == "ERASE") {
      Serial.println("Erasing chip (~30s)...");
      flash.eraseChip();
      delay(30000);
      meta.magic       = 0xDEADBEEF;
      meta.writeAddr   = DATA_START_ADDR;
      meta.totalChunks = 0;
      flashWriteAddr   = DATA_START_ADDR;
      epochIndex       = 0;
      lastErasedDataSector = 0xFFFFFFFF;
      saveMetadata();
      Serial.println("✓ Done - fresh start");
    }
  }

  // ================================================================
  // MPU @ 10Hz - 4-method combined vibration
  // ================================================================
  static uint32_t lastMPU = 0;
  if (millis() - lastMPU >= 100) {
    lastMPU = millis();

    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);

    // Method 1: accel delta between samples (removes gravity)
    float accelDelta = abs(a.acceleration.x - lastAccelX) +
                       abs(a.acceleration.y - lastAccelY) +
                       abs(a.acceleration.z - lastAccelZ);
    lastAccelX = a.acceleration.x;
    lastAccelY = a.acceleration.y;
    lastAccelZ = a.acceleration.z;

    // Method 2: gyroscope (rotation - best for bed frame)
    float gyroMag = sqrt(sq(g.gyro.x) + sq(g.gyro.y) + sq(g.gyro.z));

    // Method 3: total accel vs gravity (vertical impacts)
    float accelMag = sqrt(sq(a.acceleration.x) + sq(a.acceleration.y) + sq(a.acceleration.z));
    float accelVsGravity = abs(accelMag - 9.81);

    // Weighted combined score (gyro 3x - most useful for bed)
    float score = (accelDelta * 1.0) + (gyroMag * 3.0) + (accelVsGravity * 0.5);

    // Method 4: RMS smoothing over 10 samples
    rmsWindow[rmsIndex] = score;
    rmsIndex = (rmsIndex + 1) % RMS_WINDOW;
    float rmsSum = 0;
    for (uint8_t i = 0; i < RMS_WINDOW; i++) rmsSum += rmsWindow[i] * rmsWindow[i];
    float rmsScore = sqrt(rmsSum / RMS_WINDOW);

    vibTotalSamples++;
    vibScoreSum += rmsScore;
    if      (rmsScore > VIB_LARGE_THRESHOLD) vibLargeCount++;
    else if (rmsScore > VIB_MINOR_THRESHOLD) vibMinorCount++;

    static uint32_t lastVibDebug = 0;
    if (millis() - lastVibDebug >= 5000) {
      lastVibDebug = millis();
      Serial.print("VIB rms="); Serial.print(rmsScore, 3);
      Serial.print(" accΔ=");   Serial.print(accelDelta, 3);
      Serial.print(" gyro=");   Serial.print(gyroMag, 3);
      Serial.print(" L=");      Serial.print(vibLargeCount);
      Serial.print(" M=");      Serial.println(vibMinorCount);
    }
  }

  // ================================================================
  // RADAR @ 1Hz
  // ================================================================
  static uint32_t lastRadar = 0;
  if (millis() - lastRadar >= 1000) {
    lastRadar = millis();

    int hr = radar.getHeartRate();
    int rr = radar.getBreatheValue();

    if (hr > 30 && hr < 200 && bufferIndex < MAX_SAMPLES_PER_EPOCH) {
      hrBuffer[bufferIndex] = (uint8_t)hr;
      rrBuffer[bufferIndex] = (uint8_t)rr;
      sumHR += hr;
      sumRR += rr;
      bufferIndex++;
    }

    if (radar.smSleepData(radar.eInOrNotInBed) == 1) sumInBed++;
    radarSamples++;
  }

  // ================================================================
  // EPOCH every 30 seconds
  // ================================================================
  if (millis() - epochStartTime >= (uint32_t)(EPOCH_SECONDS * 1000)) {
    Serial.print("\n>>> Epoch "); Serial.print(epochIndex + 1);
    Serial.print("/"); Serial.println(CHUNK_EPOCHS);
    processEpoch();
    epochStartTime = millis();
  }
}

// ================================================================
// PROCESS EPOCH
// ================================================================
void processEpoch() {
  EpochQ15 ep;

  ep.hr_mean    = bufferIndex > 0 ? (uint8_t)(sumHR / bufferIndex) : 0;
  ep.rr_mean    = bufferIndex > 0 ? (uint8_t)(sumRR / bufferIndex) : 0;
  ep.hr_std     = (uint8_t)calculateStdDev(hrBuffer, bufferIndex, ep.hr_mean);
  ep.rr_std     = (uint8_t)calculateStdDev(rrBuffer, bufferIndex, ep.rr_mean);
  ep.dhr        = (int8_t)constrain((int)ep.hr_mean - (int)prevHR, -128, 127);
  ep.drr        = (int8_t)constrain((int)ep.rr_mean - (int)prevRR, -128, 127);
  ep.in_bed_pct = (uint8_t)((sumInBed * 100) / (radarSamples > 0 ? radarSamples : 1));

  uint32_t total    = vibTotalSamples > 0 ? vibTotalSamples : 1;
  ep.large_move_pct = (uint8_t)constrain((vibLargeCount * 100) / total, 0, 100);
  ep.minor_move_pct = (uint8_t)constrain((vibMinorCount * 100) / total, 0, 100);
  ep.vib_move_pct   = (uint8_t)constrain(ep.large_move_pct + ep.minor_move_pct, 0, 100);

  float avgVib  = vibTotalSamples > 0 ? vibScoreSum / vibTotalSamples : 0;
  ep.vib_resp_q = (uint8_t)constrain(100 - (int)(avgVib * 20), 0, 100);

  // getSleepComposite() - single call for all sleep data (verified from header)
  sSleepComposite sc = radar.getSleepComposite();

  ep.turnovers_delta = sc.turnoverNumber >= lastTurnovers ? sc.turnoverNumber - lastTurnovers : 0;
  ep.apnea_delta     = sc.apneaEvents    >= lastApnea     ? sc.apneaEvents    - lastApnea     : 0;
  lastTurnovers      = sc.turnoverNumber;
  lastApnea          = sc.apneaEvents;
  ep.flags           = sc.sleepState;

  ep.agree_flags = 0;
  if (ep.flags > 0 && ep.vib_move_pct > 20)       ep.agree_flags |= 0x01;
  if (ep.in_bed_pct < 50 && ep.vib_move_pct > 10) ep.agree_flags |= 0x02;
  if (radarSamples < 20)                           ep.agree_flags |= 0x04;
  if (bufferIndex < 10)                            ep.agree_flags |= 0x08;

  Serial.print("    HR=");    Serial.print(ep.hr_mean);
  Serial.print(" RR=");       Serial.print(ep.rr_mean);
  Serial.print(" InBed=");    Serial.print(ep.in_bed_pct);    Serial.print("%");
  Serial.print(" LgMv=");     Serial.print(ep.large_move_pct); Serial.print("%");
  Serial.print(" MnMv=");     Serial.print(ep.minor_move_pct); Serial.print("%");
  Serial.print(" Turns=");    Serial.print(ep.turnovers_delta);
  Serial.print(" Apnea=");    Serial.print(ep.apnea_delta);
  Serial.print(" Sleep=");    Serial.print(ep.flags);
  Serial.print(" Qual=");     Serial.println(ep.vib_resp_q);

  currentChunk.epochs[epochIndex++] = ep;
  prevHR = ep.hr_mean;
  prevRR = ep.rr_mean;

  bufferIndex = 0; sumHR = 0; sumRR = 0;
  sumInBed = 0; radarSamples = 0;
  vibLargeCount = 0; vibMinorCount = 0;
  vibTotalSamples = 0; vibScoreSum = 0;

  if (epochIndex >= CHUNK_EPOCHS) {
    Serial.println("\n╔══════════════════════════════════╗");
    Serial.println("║  CHUNK FULL - SAVING TO FLASH    ║");
    Serial.println("╚══════════════════════════════════╝");
    saveChunkToFlash();
    epochIndex = 0;
  } else {
    Serial.print("    Progress: "); Serial.print(epochIndex);
    Serial.print("/"); Serial.println(CHUNK_EPOCHS);
  }
}

// ================================================================
// SAVE CHUNK TO FLASH
// Chunk data goes to sector 2+ (never touches metadata sector).
// Metadata always erased+rewritten via saveMetadata().
// ================================================================
void saveChunkToFlash() {
  currentChunk.header.timestamp  = rtc.now().unixtime() - (EPOCH_SECONDS * CHUNK_EPOCHS);
  currentChunk.header.num_epochs = CHUNK_EPOCHS;
  currentChunk.header.crc16      = calculateCRC16((uint8_t*)&currentChunk.epochs, sizeof(currentChunk.epochs));

  Serial.print("Data addr:  "); Serial.println(flashWriteAddr);
  Serial.print("Timestamp:  "); Serial.println(currentChunk.header.timestamp);

  // Erase data sector only when first entering it
  uint32_t sectorAddr = (flashWriteAddr / 4096) * 4096;
  if (sectorAddr != lastErasedDataSector) {
    uint8_t testByte = flash.readByte(flashWriteAddr);
    if (testByte != 0xFF) {
      Serial.print("Erasing data sector: "); Serial.println(sectorAddr);
      flash.eraseSector(sectorAddr);
      delay(100);
    }
    lastErasedDataSector = sectorAddr;
  }

  bool writeOK = flash.writeByteArray(flashWriteAddr, (uint8_t*)&currentChunk, sizeof(DataChunk));

  if (writeOK) {
    // Verify
    uint8_t verifyBuf[sizeof(DataChunk)];
    flash.readByteArray(flashWriteAddr, verifyBuf, sizeof(DataChunk));
    DataChunk* v = (DataChunk*)verifyBuf;

    if (v->header.timestamp == currentChunk.header.timestamp &&
        v->header.crc16     == currentChunk.header.crc16) {

      flashWriteAddr += sizeof(DataChunk);
      meta.writeAddr  = flashWriteAddr;
      meta.totalChunks++;

      // THE FIX: erase sector 1 first, then write fresh metadata
      saveMetadata();

      Serial.print("✓✓✓ CHUNK #"); Serial.print(meta.totalChunks);
      Serial.print(" SAVED (");
      Serial.print(meta.totalChunks * 10.0 / 60.0, 1);
      Serial.println(" hours) ✓✓✓");
    } else {
      Serial.println("✗ Verification FAILED");
    }
  } else {
    Serial.println("✗✗✗ WRITE FAILED ✗✗✗");
  }
  Serial.println("══════════════════════════════════\n");
}

// ================================================================
// HELPERS
// ================================================================
void printStatus() {
  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║       SYSTEM STATUS              ║");
  Serial.println("╚══════════════════════════════════╝");
  Serial.print("Time:         "); Serial.println(rtc.now().timestamp());
  Serial.print("Chunks saved: "); Serial.println(meta.totalChunks);
  Serial.print("Recorded:     "); Serial.print(meta.totalChunks * 10.0 / 60.0, 1); Serial.println(" hours");
  Serial.print("Epoch:        "); Serial.print(epochIndex); Serial.print("/"); Serial.println(CHUNK_EPOCHS);
  Serial.print("Flash addr:   "); Serial.print(flashWriteAddr); Serial.print(" / "); Serial.println(flash.getCapacity());
  Serial.print("In bed:       "); Serial.println(radar.smSleepData(radar.eInOrNotInBed) == 1 ? "YES" : "NO");
  Serial.print("HR:           "); Serial.print(radar.getHeartRate()); Serial.println(" BPM");
  Serial.print("RR:           "); Serial.print(radar.getBreatheValue()); Serial.println(" breaths/min");
  sSleepComposite sc = radar.getSleepComposite();
  Serial.print("Sleep state:  "); Serial.println(sc.sleepState);
  Serial.print("Turnovers:    "); Serial.println(sc.turnoverNumber);
  Serial.print("Apnea events: "); Serial.println(sc.apneaEvents);
  Serial.println("══════════════════════════════════\n");
}

void handleTimeSync(String cmd) {
  int y, mo, d, h, m, s;
  if (sscanf(cmd.c_str(), "SET_TIME %d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &m, &s) == 6) {
    rtc.adjust(DateTime(y, mo, d, h, m, s));
    Serial.println("✓ Time set");
    Serial.print("  "); Serial.println(rtc.now().timestamp());
  } else {
    Serial.println("✗ Use: SET_TIME YYYY-MM-DD HH:MM:SS");
  }
}
