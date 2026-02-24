#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <Arduino.h>

void setupWiFi(bool force_config);
void checkWiFiConnection();
bool isWiFiConnected();

#endif // WIFI_MANAGER_H
