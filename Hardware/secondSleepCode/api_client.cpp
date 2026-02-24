#include "api_client.h"
#include <Preferences.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "wifi_manager.h"
#include "rgb_led.h"

// External functions to be implemented in the main .ino file
extern void update_upload_status(uint32_t chunkIndex, bool sent);
extern bool get_upload_status(uint32_t chunkIndex);
extern uint32_t get_total_chunks();
extern bool read_chunk_from_flash(uint32_t chunkIndex, DataChunk* chunk);
extern void set_system_error_state(const char* reason);


Preferences preferences;
String api_endpoint;
String api_key;
String device_id;

void setupAPIClient() {
  preferences.begin("noctis-api", false);
  api_endpoint = preferences.getString("endpoint", "http://192.168.1.100:5000/api/epochs:ingest-device");
  api_key = preferences.getString("apikey", "");
  
  // Generate and store device ID from MAC address if not already set
  device_id = preferences.getString("device_id", "");
  if (device_id.length() == 0) {
    String mac = WiFi.macAddress();
    mac.replace(":", "");
    mac.toLowerCase();
    device_id = "noctis_" + mac;
    preferences.putString("device_id", device_id);
  }
  Serial.print("Device ID: ");
  Serial.println(device_id);
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

String serializeChunkToJSON(DataChunk* chunk) {
  StaticJsonDocument<2048> doc; // Adjusted size for a batch of epochs

  doc["device_external_id"] = device_id;
  
  JsonArray epochs = doc.createNestedArray("epochs");

  for (int i = 0; i < chunk->header.num_epochs; i++) {
    JsonObject epoch = epochs.createNestedObject();
    
    // Convert chunk timestamp + epoch index to epoch timestamp
    time_t epoch_ts = chunk->header.timestamp + (i * 30); // 30s per epoch
    char iso_ts[21];
    strftime(iso_ts, sizeof(iso_ts), "%Y-%m-%dT%H:%M:%SZ", gmtime(&epoch_ts));
    
    epoch["epoch_start_ts"] = iso_ts;
    epoch["epoch_index"] = i; // This is the index within the chunk, might need adjustment for absolute index

    JsonObject metrics = epoch.createNestedObject("metrics");
    EpochQ15* ep = &chunk->epochs[i];
    metrics["in_bed_pct"] = ep->in_bed_pct;
    metrics["hr_mean"] = ep->hr_mean;
    metrics["hr_std"] = ep->hr_std;
    metrics["dhr"] = ep->dhr;
    metrics["rr_mean"] = ep->rr_mean;
    metrics["rr_std"] = ep->rr_std;
    metrics["drr"] = ep->drr;
    metrics["large_move_pct"] = ep->large_move_pct;
    metrics["minor_move_pct"] = ep->minor_move_pct;
    metrics["turnovers_delta"] = ep->turnovers_delta;
    metrics["apnea_delta"] = ep->apnea_delta;
    metrics["flags"] = ep->flags;
    metrics["vib_move_pct"] = ep->vib_move_pct;
    metrics["vib_resp_q"] = ep->vib_resp_q;
    metrics["agree_flags"] = ep->agree_flags;
  }

  String output;
  serializeJson(doc, output);
  return output;
}

bool uploadChunk(DataChunk* chunk, uint32_t chunkIndex) {
  if (!isWiFiConnected()) {
    return false;
  }

  HTTPClient http;
  http.begin(api_endpoint);
  http.addHeader("Content-Type", "application/json");
  if (api_key.length() > 0) {
    http.addHeader("Authorization", "Bearer " + api_key);
  }
  
  String jsonPayload = serializeChunkToJSON(chunk);
  
  Serial.print("Uploading chunk #");
  Serial.print(chunkIndex);
  Serial.print("... ");

  int httpCode = http.POST(jsonPayload);

  if (httpCode > 0) {
    Serial.printf("HTTP %d\n", httpCode);
    if (httpCode == HTTP_CODE_OK) {
      update_upload_status(chunkIndex, true);
      http.end();
      return true;
    } else {
      String payload = http.getString();
      Serial.println(payload);
    }
  } else {
    Serial.printf("Upload failed, error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  return false;
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
                if (uploadChunk(&chunk, i)) {
                    successCount++;
                } else {
                    // Stop on first failure to avoid hammering a broken endpoint
                    Serial.println("Upload failed, pausing backlog.");
                    led_offline_mode();
                    return;
                }
            } else {
                Serial.print("Failed to read chunk #");
                Serial.println(i);
                set_system_error_state("Flash read failed");
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
    Serial.print("Unsent Chunks:  "); Serial.println(getBacklogCount());
    Serial.println("------------------\n");
}
