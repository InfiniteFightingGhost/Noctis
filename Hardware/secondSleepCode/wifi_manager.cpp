#include "wifi_manager.h"
#include "rgb_led.h"
#include <WiFi.h>
#include <WiFiManager.h>
#include <esp_task_wdt.h>

WiFiManager wm;
bool wifi_connected = false;

void setupWiFi(bool force_config) {
  led_wifi_connecting();
  
  // Set custom AP name
  wm.setAPCallback([](WiFiManager *myWiFiManager) {
    Serial.println("Entered config mode");
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());
  });

  // Set a timeout for configuration
  wm.setConfigPortalTimeout(180); // 3 minutes

  // Disable WDT during blocking WiFi setup
  esp_task_wdt_delete(NULL);

  bool connected;
  if (force_config) {
    connected = wm.startConfigPortal("Noctis-Setup");
  } else {
    connected = wm.autoConnect("Noctis-Setup");
  }

  // Re-enable WDT
  esp_task_wdt_add(NULL);

  if (!connected) {
      Serial.println("Failed to connect and hit timeout");
      // Go into offline mode
      led_offline_mode();
      wifi_connected = false;
      return;
  }

  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  led_connected_blinks();
  wifi_connected = true;
}

void checkWiFiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    if (wifi_connected) {
      Serial.println("WiFi disconnected. Reconnecting...");
      wifi_connected = false;
      led_offline_mode();
    }
    // Try to reconnect, disabling WDT for the blocking call
    esp_task_wdt_delete(NULL);
    bool reconnected = wm.autoConnect("Noctis-Setup");
    esp_task_wdt_add(NULL);
    
    if (reconnected) {
        Serial.println("WiFi reconnected!");
        wifi_connected = true;
        led_connected_blinks();
    }
  }
}

bool isWiFiConnected() {
  return wifi_connected && (WiFi.status() == WL_CONNECTED);
}
