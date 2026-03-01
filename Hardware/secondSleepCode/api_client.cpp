#include "api_client.h"
#include <Preferences.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "wifi_manager.h"
#include "rgb_led.h"
#include "RTClib.h"

// External functions and globals from the main .ino file
extern void update_upload_status(uint32_t chunkIndex, bool sent);
extern bool get_upload_status(uint32_t chunkIndex);
extern uint32_t get_total_chunks();
extern bool read_chunk_from_flash(uint32_t chunkIndex, DataChunk* chunk);
extern void set_system_error_state(const char* reason);
extern FlashMetadata meta; 

// External globals defined in the main .ino
extern RTC_DS3231 rtc;


Preferences preferences;
String api_endpoint;
String api_key;
String device_id;
String recording_id_str; // In-memory cache of the current recording ID

// Helper to format unix time to ISO8601 string
void toISO8601(uint32_t unix_time, char* buffer, size_t buffer_size) {
    DateTime dt(unix_time);
    snprintf(buffer, buffer_size, "%04d-%02d-%02dT%02d:%02d:%02dZ", 
             dt.year(), dt.month(), dt.day(), dt.hour(), dt.minute(), dt.second());
}

void setupAPIClient() {
  preferences.begin("noctis-api", false);
  api_endpoint = preferences.getString("endpoint", "https://backend-production-2e2e.up.railway.app/v1/epochs:ingest-device");
  api_key = preferences.getString("apikey", "");
  
  device_id = preferences.getString("device_id", "noctis-halo-s1-001");
  preferences.putString("device_id", device_id);

  Serial.print("Device External ID: ");
  Serial.println(device_id);
}

void setRecordingID(String id) {
  recording_id_str = id;
}

String startNewRecording() {
  if (!isWiFiConnected()) {
    Serial.println("Cannot start recording, WiFi not connected.");
    return "";
  }

  HTTPClient http;
  String start_url = api_endpoint;
  // Derive the start-recording URL from the ingest URL
  start_url.replace("epochs:ingest-device", "recordings:start");
  
  http.begin(start_url);
  http.addHeader("Content-Type", "application/json");
  // This endpoint uses API Key authentication.
  if (api_key.length() > 0) {
    http.addHeader("X-API-Key", api_key);
  }

  StaticJsonDocument<256> doc;
  doc["device_external_id"] = device_id;
  doc["timezone"] = "UTC";

  String payload;
  serializeJson(doc, payload);

  Serial.print("Starting new recording session... ");
  int httpCode = http.POST(payload);

  String result_id = "";
  if (httpCode == HTTP_CODE_OK) {
    String response = http.getString();
    StaticJsonDocument<512> respDoc;
    DeserializationError error = deserializeJson(respDoc, response);
    if (!error) {
      result_id = respDoc["id"].as<String>();
      setRecordingID(result_id);
      Serial.println("OK");
      Serial.print("New ID: "); Serial.println(result_id);
    } else {
      Serial.println("Failed to parse response.");
    }
  } else {
    Serial.printf("Failed. HTTP %d\n", httpCode);
    if (httpCode > 0) Serial.println(http.getString());
  }

  http.end();
  return result_id;
}


void setAPIEndpoint(String endpoint) {
  api_endpoint = endpoint;
  preferences.putString("endpoint", endpoint);
  Serial.print("API endpoint set to: ");
  Serial.println(endpoint);
}

void setAPIKey(String key) {
  api_key = key;
  preferences.putString("apikey", key);
  Serial.println("API key set.");
}

String getAPIEndpoint() {
  return api_endpoint;
}

String getAPIKey() {
  return api_key;
}

void pingAPI() {
  if (!isWiFiConnected()) {
    Serial.println("Cannot ping API, WiFi not connected.");
    return;
  }
  HTTPClient http;
  http.begin(api_endpoint);
  Serial.print("Pinging API endpoint... ");
  int httpCode = http.GET();
  if (httpCode > 0) {
    Serial.printf("OK. HTTP Code: %d\n", httpCode);
    String payload = http.getString();
    Serial.println("Response payload:");
    Serial.println(payload);
  } else {
    Serial.printf("Failed. Error: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

bool uploadChunkAsEpochBatch(DataChunk* chunk, uint32_t chunkIndex) {
    if (!isWiFiConnected()) {
        Serial.println("Cannot upload: WiFi not connected.");
        return false;
    }

    DynamicJsonDocument doc(16384);

    doc["device_external_id"] = device_id;
    doc["device_name"] = "Sleep Monitor v4.0 WiFi";
    doc["forward_to_ml"] = true;

    char iso_buffer[21];
    toISO8601(meta.recording_start_ts, iso_buffer, sizeof(iso_buffer));
    doc["recording_started_at"] = iso_buffer;

    JsonArray epochs = doc.createNestedArray("epochs");

    for (int i = 0; i < chunk->header.num_epochs; i++) {
        EpochQ15* ep = &chunk->epochs[i];
        JsonObject epoch_obj = epochs.createNestedObject();

        epoch_obj["epoch_index"] = chunkIndex * CHUNK_EPOCHS + i;

        uint32_t epoch_unix_time = chunk->header.timestamp + (i * EPOCH_SECONDS);
        toISO8601(epoch_unix_time, iso_buffer, sizeof(iso_buffer));
        epoch_obj["epoch_start_ts"] = iso_buffer;

        JsonArray metrics = epoch_obj.createNestedArray("metrics");
        metrics.add(ep->in_bed_pct / 100.0f);
        metrics.add((float)ep->hr_mean);
        metrics.add((float)ep->hr_std);
        metrics.add((float)ep->dhr);
        metrics.add((float)ep->rr_mean);
        metrics.add((float)ep->rr_std);
        metrics.add((float)ep->drr);
        metrics.add(ep->large_move_pct / 100.0f);
        metrics.add(ep->minor_move_pct / 100.0f);
        metrics.add((float)ep->turnovers_delta);
        metrics.add((float)ep->apnea_delta);
        metrics.add((float)ep->flags);
        metrics.add(ep->vib_move_pct / 100.0f);
        metrics.add(ep->vib_resp_q / 100.0f);
        metrics.add(ep->agree_flags / 100.0f);
    }
    
    String jsonPayload;
    serializeJson(doc, jsonPayload);

    HTTPClient http;
    http.begin(api_endpoint);
    http.addHeader("Content-Type", "application/json");
    if (api_key.length() > 0) {
        http.addHeader("X-API-Key", api_key);
    }
    
    Serial.print("Uploading chunk #");
    Serial.print(chunkIndex);
    Serial.print(" as epoch batch... ");
    
    int httpCode = http.POST(jsonPayload);

    if (httpCode == HTTP_CODE_OK) {
        Serial.printf("HTTP 200 OK\n");
        http.end();
        return true;
    } else {
        Serial.printf("HTTP %d Error\n", httpCode);
        String responseBody = http.getString();
        Serial.println("Response body:");
        Serial.println(responseBody);
        http.end();
        return false;
    }
}

uint32_t getBacklogCount() {
    uint32_t total = get_total_chunks();
    uint32_t count = 0;
    for (uint32_t i = 0; i < total; i++) {
        if (!get_upload_status(i)) {
            count++;
        }
    }
    return count;
}

void uploadBacklog() {
    if (!isWiFiConnected()) {
        Serial.println("Cannot upload backlog, WiFi not connected.");
        return;
    }

    led_uploading();
    Serial.println("Starting backlog upload...");
    
    uint32_t totalChunks = get_total_chunks();
    if (totalChunks == 0) {
        Serial.println("No chunks to upload.");
        led_off();
        return;
    }

    uint32_t successCount = 0;
    for (uint32_t i = 0; i < totalChunks; i++) {
        if (!get_upload_status(i)) {
            DataChunk chunk;
            if (read_chunk_from_flash(i, &chunk)) {
                if (uploadChunkAsEpochBatch(&chunk, i)) {
                    update_upload_status(i, true);
                    successCount++;
                } else {
                    Serial.println("Upload failed, pausing backlog.");
                    led_offline_mode();
                    return;
                }
            } else {
                char reason[50];
                sprintf(reason, "Flash read failed for chunk %lu", i);
                set_system_error_state(reason);
                return;
            }
        }
    }
    Serial.print("Backlog upload finished. Uploaded ");
    Serial.print(successCount);
    Serial.println(" chunks.");
    led_off();
}

void printAPIStatus() {
    Serial.println("\n--- API STATUS ---");
    Serial.print("WiFi Connected: "); Serial.println(isWiFiConnected() ? "Yes" : "No");
    Serial.print("API Endpoint:   "); Serial.println(api_endpoint);
    Serial.print("API Key Set:    "); Serial.println(api_key.length() > 0 ? "Yes" : "No");
    Serial.print("Device ID:      "); Serial.println(device_id);
    Serial.print("Recording ID:   "); Serial.println(recording_id_str);
    Serial.print("Unsent Chunks:  "); Serial.println(getBacklogCount());
    Serial.println("------------------\n");
}
