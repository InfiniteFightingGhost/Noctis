/*
 * ESP32 Sleep Monitor - v4.0 (WiFi Edition)
 *
 * v4.0: Adds WiFi connectivity, API uploads, and status RGB LED.
 * Flash metadata is updated to v2 to track upload status.
 *
 * Flash layout:
 *   Sector 0  (0x0000-0x0FFF): unused / reserved
 *   Sector 1  (0x1000-0x1FFF): metadata ONLY - erased on every update
 *   Sector 2+ (0x2000+):       chunk data, never erased unless full
 */

// --- Core Libraries ---
#include <Wire.h>
#include <SPI.h>
#include <RTClib.h>
#include <SPIMemory.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DFRobot_HumanDetection.h>
#include <esp_task_wdt.h>

// Undefine conflicting macro from SPIMemory library before including WiFi headers
#undef ID

// --- WiFi and API Libraries ---
#include <WiFi.h>
#include <WiFiManager.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Preferences.h>

// --- Project Headers ---
#include "types.h"
#include "rgb_led.h"
#include "wifi_manager.h"
#include "api_client.h"


// --- PIN DEFINITIONS ---
#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   5
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   19
#define PIN_I2C_SDA    2
#define PIN_I2C_SCL    15
#define PIN_C1001_RX   27
#define PIN_C1001_TX   26

// --- FLASH LAYOUT ---
#define METADATA_ADDR   0x1000   // Sector 1 - metadata only
#define DATA_START_ADDR 0x2000   // Sector 2 - all chunk data starts here

// --- CONFIGURATION ---
#define MPU_ADDR              0x69
#define MAX_SAMPLES_PER_EPOCH 35
#define WDT_TIMEOUT_S         30

// --- VIBRATION THRESHOLDS ---
#define VIB_LARGE_THRESHOLD 0.8
#define VIB_MINOR_THRESHOLD 0.15


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
bool system_in_error_state = false;


// --- FORWARD DECLARATIONS ---
void     saveMetadata();
void     printStatus();
void     handleTimeSync(String cmd);
void     processEpoch();
void     saveChunkToFlash();
float    calculateStdDev(uint8_t* data, uint8_t count, uint8_t mean);
uint16_t calculateCRC16(uint8_t* data, uint16_t length);
void     initializeNewMetadata();
void     handleMetadataMigration();
void     set_system_error_state(const char* reason);


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

void set_system_error_state(const char* reason) {
  Serial.print("\n\n!!! SYSTEM HALTED: ");
  Serial.println(reason);
  system_in_error_state = true;
  led_error_solid();
  while(1) {
    esp_task_wdt_reset();
    delay(1000);
  }
}

void indicateError(uint8_t code) {
  char buffer[50];
  sprintf(buffer, "Legacy error code: %d", code);
  set_system_error_state(buffer);
}

void saveMetadata() {
  flash.eraseSector(METADATA_ADDR);
  delay(50); // Give erase time to complete
  if (!flash.writeByteArray(METADATA_ADDR, (uint8_t*)&meta, sizeof(FlashMetadata))) {
    set_system_error_state("Failed to save metadata!");
  }
}

void initializeNewMetadata() {
    Serial.println("✓ Fresh start - initializing v3 metadata");
    meta.version = 3;
    meta.magic = 0xDEADBEEF;
    meta.writeAddr = DATA_START_ADDR;
    meta.totalChunks = 0;
    meta.uploadedChunks = 0;
    meta.recording_start_ts = rtc.now().unixtime();
    memset(meta.recording_id, 0, sizeof(meta.recording_id));
    memset(meta.upload_bitmap, 0, UPLOAD_BITMAP_SIZE);
    flashWriteAddr = DATA_START_ADDR;
    saveMetadata();
    Serial.println("✓ Metadata written to sector 1 (0x1000)");
}

void handleMetadataMigration() {
    flash.readAnything(METADATA_ADDR, meta);

    if (meta.magic == 0xDEADBEEF && meta.version == 3) {
        // V3 metadata found, all good
        flashWriteAddr = meta.writeAddr;
        setRecordingID(String(meta.recording_id));
        Serial.print("✓ Resuming v3 - chunks saved: "); Serial.println(meta.totalChunks);
        Serial.print("  Recording ID: "); Serial.println(meta.recording_id);
    } else {
        // Check for V2
        FlashMetadataV2 meta_v2;
        flash.readAnything(METADATA_ADDR, meta_v2);
        if (meta_v2.magic == 0xDEADBEEF && meta_v2.version == 2) {
            Serial.println("!!! Old v2 metadata found. Upgrading to v3...");
            meta.version = 3;
            meta.magic = 0xDEADBEEF;
            meta.writeAddr = meta_v2.writeAddr;
            meta.totalChunks = meta_v2.totalChunks;
            meta.uploadedChunks = meta_v2.uploadedChunks;
            meta.recording_start_ts = rtc.now().unixtime(); // Set new field to now
            memset(meta.recording_id, 0, sizeof(meta.recording_id)); // Clear recording ID to force new one
            memcpy(meta.upload_bitmap, meta_v2.upload_bitmap, UPLOAD_BITMAP_SIZE);
            flashWriteAddr = meta.writeAddr;
            saveMetadata();
            Serial.println("✓ Metadata migrated to v3.");
        } else {
            // Check for V1
            FlashMetadataV1 meta_v1;
            flash.readAnything(METADATA_ADDR, meta_v1);
            if (meta_v1.magic == 0xDEADBEEF) {
                Serial.println("!!! Old v1 metadata found. Upgrading to v3...");
                meta.version = 3;
                meta.magic = 0xDEADBEEF;
                meta.writeAddr = meta_v1.writeAddr;
                meta.totalChunks = meta_v1.totalChunks;
                meta.uploadedChunks = 0;
                meta.recording_start_ts = rtc.now().unixtime(); // Set new field to now
                memset(meta.recording_id, 0, sizeof(meta.recording_id)); // Clear recording ID
                memset(meta.upload_bitmap, 0, UPLOAD_BITMAP_SIZE);
                flashWriteAddr = meta.writeAddr;
                saveMetadata();
                Serial.println("✓ Metadata migrated to v3.");
            } else {
                // No valid metadata, fresh start
                initializeNewMetadata();
            }
        }
    }
}


// --- SETUP ---
void setup() {
  Serial.begin(115200);
  Serial2.begin(115200, SERIAL_8N1, PIN_C1001_RX, PIN_C1001_TX);
  
  setupRGB();

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.setClock(400000);
  SPI.begin(PIN_SPI_SCK, PIN_SPI_MISO, PIN_SPI_MOSI, PIN_FLASH_CS);
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE0);

  esp_task_wdt_init(WDT_TIMEOUT_S, true);
  esp_task_wdt_add(NULL);

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║   Sleep Monitor v4.0 (WiFi)      ║");
  Serial.println("╚══════════════════════════════════╝");

  if (!rtc.begin())             { set_system_error_state("RTC fail!"); }
  Serial.println("✓ RTC OK");

  if (!mpu.begin(MPU_ADDR, &Wire)) { set_system_error_state("MPU fail!"); }
  Serial.println("✓ MPU6050 OK");
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  if (!flash.begin())           { set_system_error_state("Flash fail!"); }
  Serial.print("✓ Flash OK - ");
  Serial.print(flash.getCapacity()); Serial.println(" bytes");

  handleMetadataMigration();
  
  radar.begin();
  radar.configWorkMode(radar.eSleepMode);
  delay(500);
  Serial.println("✓ Radar OK");

  // WiFi and API Setup
  led_wifi_connecting();
  setupWiFi(false); // false = don't force config portal
  setupAPIClient();

  if(isWiFiConnected()) {
    led_connected_blinks();
    
    // Explicit session start if no current recording ID exists in Flash
    if (strlen(meta.recording_id) == 0) {
      Serial.println("No recording session found in Flash. Requesting from backend...");
      String new_id = startNewRecording();
      if (new_id.length() > 0) {
        strncpy(meta.recording_id, new_id.c_str(), 36);
        meta.recording_id[36] = '\0';
        saveMetadata();
      } else {
        Serial.println("⚠ Failed to start recording session. Will retry later or use local ID.");
      }
    } else {
      Serial.print("Using existing recording session: "); Serial.println(meta.recording_id);
      setRecordingID(String(meta.recording_id));
    }
  } else {
    led_offline_mode();
    if (strlen(meta.recording_id) > 0) {
      setRecordingID(String(meta.recording_id));
    }
  }

  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║      RECORDING STARTED           ║");
  Serial.println("╚══════════════════════════════════╝");
  Serial.println("Epoch every 30s | Chunk every 10min");
  Serial.println("Commands: STATUS | DEBUG | ERASE | SET_TIME | WIFI_STATUS | API_STATUS | UPLOAD_NOW | etc.");
  Serial.println("══════════════════════════════════\n");

  epochStartTime = millis();
}

// --- MAIN LOOP ---
void loop() {
  esp_task_wdt_reset();

  if(system_in_error_state) {
    delay(1000);
    return;
  }

  // Handle Serial Commands
  if (Serial.available()) {
    String raw_cmd = Serial.readStringUntil('\n');
    raw_cmd.trim();
    String cmd_upper = raw_cmd;
    cmd_upper.toUpperCase();

    if (cmd_upper == "STATUS") {
      printStatus();
    } else if (cmd_upper.startsWith("SET_TIME ")) {
      handleTimeSync(cmd_upper);
    } else if (cmd_upper == "DEBUG") {
      Serial.println("\n--- DEBUG ---");
      Serial.print("epochIndex:     "); Serial.print(epochIndex); Serial.print("/"); Serial.println(CHUNK_EPOCHS);
      Serial.print("Next epoch in:  "); Serial.print((EPOCH_SECONDS*1000-(millis()-epochStartTime))/1000); Serial.println("s");
      Serial.print("flashWriteAddr: "); Serial.println(flashWriteAddr);
      Serial.print("totalChunks:    "); Serial.println(meta.totalChunks);
      Serial.println("-------------\n");
    } else if (cmd_upper == "ERASE") {
      Serial.println("!!! This will erase all data. Type 'CONFIRM' to proceed.");
      Serial.setTimeout(10000); // Increase timeout to 10s for confirmation
      String confirm = Serial.readStringUntil('\n');
      confirm.trim();
      Serial.setTimeout(1000); // Reset timeout to default
      if(confirm == "CONFIRM") {
        Serial.println("Erasing chip...");
        flash.eraseChip();
        delay(30000);
        initializeNewMetadata();
        epochIndex = 0;
        lastErasedDataSector = 0xFFFFFFFF;
        Serial.println("✓ Done - fresh start");
      } else {
        Serial.println("Cancelled.");
      }
    } else if (cmd_upper == "WIFI_STATUS") {
        Serial.println("\n--- WIFI STATUS ---");
        Serial.print("Connected: "); Serial.println(isWiFiConnected() ? "Yes" : "No");
        if(isWiFiConnected()) {
          Serial.print("IP: "); Serial.println(WiFi.localIP());
          Serial.print("RSSI: "); Serial.println(WiFi.RSSI());
        }
        Serial.println("-------------------\n");
    } else if (cmd_upper == "API_STATUS") {
        printAPIStatus();
    } else if (cmd_upper == "WIFI_RESET") {
        Serial.println("Resetting WiFi settings and restarting...");
        setupWiFi(true); // true = force config
    } else if (cmd_upper.startsWith("SET_API ")) {
        String url = raw_cmd.substring(8);
        setAPIEndpoint(url);
    } else if (cmd_upper.startsWith("SET_API_KEY ")) {
        String key = raw_cmd.substring(12);
        setAPIKey(key);
    } else if (cmd_upper == "GET_API_KEY") {
        String key = getAPIKey();
        Serial.print("Current API Key: ");
        Serial.println(key);
    } else if (cmd_upper == "PING_API") {
        pingAPI();
    } else if (cmd_upper == "UPLOAD_NOW") {
        uploadBacklog();
    } else if (cmd_upper == "WIFI_SCAN") {
        Serial.println("Scanning for WiFi networks...");
        int n = WiFi.scanNetworks();
        if (n == 0) {
            Serial.println("No networks found.");
        } else {
            Serial.print(n); Serial.println(" networks found:");
            for (int i = 0; i < n; ++i) {
                Serial.printf("  %d: %s (%d) %s\n", i + 1, WiFi.SSID(i).c_str(), WiFi.RSSI(i), (WiFi.encryptionType(i) == WIFI_AUTH_OPEN) ? " " : "*");
            }
        }
    } else if (cmd_upper == "NEW_NIGHT") {
        Serial.println("!!! Starting new night. Current session will be closed.");
        if (!isWiFiConnected()) {
            Serial.println("✗ WiFi required to start new session.");
        } else {
            String new_id = startNewRecording();
            if (new_id.length() > 0) {
                strncpy(meta.recording_id, new_id.c_str(), 36);
                meta.recording_id[36] = '\0';
                saveMetadata();
                Serial.println("✓ New session started successfully.");
            } else {
                Serial.println("✗ Failed to start new session.");
            }
        }
    }
  }

  // Maintain WiFi Connection
  static uint32_t lastWifiCheck = 0;
  if(millis() - lastWifiCheck > 30000) { // Check every 30s
    lastWifiCheck = millis();
    if(!isWiFiConnected()) {
      led_offline_mode();
    }
    checkWiFiConnection();
  }

  // ================================================================
  // MPU @ 10Hz
  // ================================================================
  static uint32_t lastMPU = 0;
  if (millis() - lastMPU >= 100) {
    lastMPU = millis();
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    float accelDelta = abs(a.acceleration.x - lastAccelX) + abs(a.acceleration.y - lastAccelY) + abs(a.acceleration.z - lastAccelZ);
    lastAccelX = a.acceleration.x; lastAccelY = a.acceleration.y; lastAccelZ = a.acceleration.z;
    float gyroMag = sqrt(sq(g.gyro.x) + sq(g.gyro.y) + sq(g.gyro.z));
    float accelMag = sqrt(sq(a.acceleration.x) + sq(a.acceleration.y) + sq(a.acceleration.z));
    float accelVsGravity = abs(accelMag - 9.81);
    float score = (accelDelta * 1.0) + (gyroMag * 3.0) + (accelVsGravity * 0.5);
    rmsWindow[rmsIndex] = score;
    rmsIndex = (rmsIndex + 1) % RMS_WINDOW;
    float rmsSum = 0;
    for (uint8_t i = 0; i < RMS_WINDOW; i++) rmsSum += rmsWindow[i] * rmsWindow[i];
    float rmsScore = sqrt(rmsSum / RMS_WINDOW);
    vibTotalSamples++;
    vibScoreSum += rmsScore;
    if (rmsScore > VIB_LARGE_THRESHOLD) vibLargeCount++;
    else if (rmsScore > VIB_MINOR_THRESHOLD) vibMinorCount++;
  }

  // ================================================================
  // RADAR @ 1Hz
  // ================================================================
  static uint32_t lastRadar = 0;
  if (millis() - lastRadar >= 1000) {
    lastRadar = millis();
    int hr = radar.getHeartRate();
    int rr = radar.getBreatheValue();
    if (bufferIndex < MAX_SAMPLES_PER_EPOCH) {
      hrBuffer[bufferIndex] = (uint8_t)hr;
      rrBuffer[bufferIndex] = (uint8_t)rr;
      bufferIndex++;
    }
    sumHR += hr; sumRR += rr;
    uint16_t movement = radar.smHumanData(radar.eHumanMovement);
    bool validVitals = (hr > 40 && hr < 180 && rr > 8 && rr < 30);
    bool inBed = validVitals && (movement < 10);
    if (inBed) sumInBed++;
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
  ep.hr_mean = bufferIndex > 0 ? (uint8_t)(sumHR / bufferIndex) : 0;
  ep.rr_mean = bufferIndex > 0 ? (uint8_t)(sumRR / bufferIndex) : 0;
  ep.hr_std = (uint8_t)calculateStdDev(hrBuffer, bufferIndex, ep.hr_mean);
  ep.rr_std = (uint8_t)calculateStdDev(rrBuffer, bufferIndex, ep.rr_mean);
  ep.dhr = (int8_t)constrain((int)ep.hr_mean - (int)prevHR, -128, 127);
  ep.drr = (int8_t)constrain((int)ep.rr_mean - (int)prevRR, -128, 127);
  ep.in_bed_pct = (uint8_t)((sumInBed * 100) / (radarSamples > 0 ? radarSamples : 1));
  uint32_t total = vibTotalSamples > 0 ? vibTotalSamples : 1;
  ep.large_move_pct = (uint8_t)constrain((vibLargeCount * 100) / total, 0, 100);
  ep.minor_move_pct = (uint8_t)constrain((vibMinorCount * 100) / total, 0, 100);
  ep.vib_move_pct = (uint8_t)constrain(ep.large_move_pct + ep.minor_move_pct, 0, 100);
  float avgVib  = vibTotalSamples > 0 ? vibScoreSum / vibTotalSamples : 0;
  ep.vib_resp_q = (uint8_t)constrain(100 - (int)(avgVib * 20), 0, 100);
  sSleepComposite sc = radar.getSleepComposite();
  ep.turnovers_delta = sc.turnoverNumber >= lastTurnovers ? sc.turnoverNumber - lastTurnovers : 0;
  ep.apnea_delta = sc.apneaEvents >= lastApnea ? sc.apneaEvents - lastApnea : 0;
  lastTurnovers = sc.turnoverNumber;
  lastApnea = sc.apneaEvents;
  ep.flags = sc.sleepState;
  ep.agree_flags = 0;
  if (ep.flags > 0 && ep.vib_move_pct > 20) ep.agree_flags |= 0x01;
  if (ep.in_bed_pct < 50 && ep.vib_move_pct > 10) ep.agree_flags |= 0x02;
  if (radarSamples < 20) ep.agree_flags |= 0x04;
  if (bufferIndex < 10) ep.agree_flags |= 0x08;

  currentChunk.epochs[epochIndex++] = ep;
  prevHR = ep.hr_mean; prevRR = ep.rr_mean;
  bufferIndex = 0; sumHR = 0; sumRR = 0;
  sumInBed = 0; radarSamples = 0;
  vibLargeCount = 0; vibMinorCount = 0;
  vibTotalSamples = 0; vibScoreSum = 0;

  if (epochIndex >= CHUNK_EPOCHS) {
    saveChunkToFlash();
    epochIndex = 0;
  }
}

// ================================================================
// SAVE CHUNK TO FLASH
// ================================================================
void saveChunkToFlash() {
  // Check if flash is full before proceeding
  if (flashWriteAddr + sizeof(DataChunk) > flash.getCapacity()) {
    // This could be made smarter, e.g. set a flag to only print once
    Serial.println("!!! Flash memory is full! Cannot save new chunk. Please ERASE.");
    return;
  }

  currentChunk.header.timestamp  = rtc.now().unixtime() - (EPOCH_SECONDS * CHUNK_EPOCHS);
  currentChunk.header.num_epochs = CHUNK_EPOCHS;
  currentChunk.header.crc16      = calculateCRC16((uint8_t*)&currentChunk.epochs, sizeof(currentChunk.epochs));

  uint32_t sectorAddr = (flashWriteAddr / 4096) * 4096;
  if (sectorAddr != lastErasedDataSector) {
    uint8_t testByte = flash.readByte(flashWriteAddr);
    if (testByte != 0xFF) {
      if (!flash.eraseSector(sectorAddr)) {
        char reason[50];
        sprintf(reason, "Flash erase failed at addr %lu", sectorAddr);
        set_system_error_state(reason);
        return;
      }
      delay(100);
    }
    lastErasedDataSector = sectorAddr;
  }

  bool writeOK = flash.writeByteArray(flashWriteAddr, (uint8_t*)&currentChunk, sizeof(DataChunk));

  if (writeOK) {
      uint32_t savedChunkIndex = meta.totalChunks;
      flashWriteAddr += sizeof(DataChunk);
      meta.writeAddr  = flashWriteAddr;
      meta.totalChunks++;
      update_upload_status(savedChunkIndex, false); // Mark as unsent
      saveMetadata();

      Serial.print("✓ CHUNK #"); Serial.print(meta.totalChunks); Serial.println(" SAVED");

      // Attempt real-time upload
      if(uploadChunkAsEpochBatch(&currentChunk, savedChunkIndex)) {
          update_upload_status(savedChunkIndex, true);
      }

  } else {
    set_system_error_state("Flash write failed!");
  }
}

// ================================================================
// HELPERS & API/FLASH BRIDGE
// ================================================================
void printStatus() {
  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║       SYSTEM STATUS              ║");
  Serial.println("╚══════════════════════════════════╝");
  Serial.print("Time:         "); Serial.println(rtc.now().timestamp());
  Serial.print("Chunks saved: "); Serial.println(meta.totalChunks);
  Serial.print("Chunks uploaded: "); Serial.print(meta.uploadedChunks); Serial.print("/"); Serial.println(meta.totalChunks);
  Serial.print("Epoch:        "); Serial.print(epochIndex); Serial.print("/"); Serial.println(CHUNK_EPOCHS);
  Serial.print("Flash addr:   "); Serial.print(flashWriteAddr); Serial.print(" / "); Serial.println(flash.getCapacity());
  Serial.print("HR:           "); Serial.print(radar.getHeartRate()); Serial.println(" BPM");
  Serial.print("RR:           "); Serial.print(radar.getBreatheValue()); Serial.println(" breaths/min");
  Serial.println("══════════════════════════════════\n");
}

void handleTimeSync(String cmd) {
  int y, mo, d, h, m, s;
  if (sscanf(cmd.c_str(), "SET_TIME %d-%d-%d %d:%d:%d", &y, &mo, &d, &h, &m, &s) == 6) {
    rtc.adjust(DateTime(y, mo, d, h, m, s));
    Serial.println("✓ Time set");
  } else {
    Serial.println("✗ Use: SET_TIME YYYY-MM-DD HH:MM:SS");
  }
}

void update_upload_status(uint32_t chunkIndex, bool sent) {
  if (chunkIndex >= (UPLOAD_BITMAP_SIZE * 8)) return; // Out of bounds
  uint32_t byte_index = chunkIndex / 8;
  uint8_t bit_index = chunkIndex % 8;
  bool was_sent = (meta.upload_bitmap[byte_index] >> bit_index) & 1;

  if (sent && !was_sent) {
    meta.upload_bitmap[byte_index] |= (1 << bit_index);
    meta.uploadedChunks++;
    saveMetadata();
  } else if (!sent && was_sent) {
    // This case should not happen often, but for completeness
    meta.upload_bitmap[byte_index] &= ~(1 << bit_index);
    meta.uploadedChunks--;
    saveMetadata();
  }
}

bool get_upload_status(uint32_t chunkIndex) {
  if (chunkIndex >= (UPLOAD_BITMAP_SIZE * 8)) return false;
  uint32_t byte_index = chunkIndex / 8;
  uint8_t bit_index = chunkIndex % 8;
  return (meta.upload_bitmap[byte_index] >> bit_index) & 1;
}

uint32_t get_total_chunks() {
  return meta.totalChunks;
}

bool read_chunk_from_flash(uint32_t chunkIndex, DataChunk* chunk) {
  if (chunkIndex >= meta.totalChunks) return false;
  uint32_t addr = DATA_START_ADDR + (chunkIndex * sizeof(DataChunk));
  return flash.readByteArray(addr, (uint8_t*)chunk, sizeof(DataChunk));
}
