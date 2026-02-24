# Noctis WiFi Serial Commands

This document lists the new serial commands added for WiFi and API functionality.

## `WIFI_STATUS`
Shows the current WiFi connection status, IP address, and signal strength (RSSI).

**Usage:**
```
WIFI_STATUS
```

## `API_STATUS`
Shows the API configuration and backlog status.

**Usage:**
```
API_STATUS
```
**Output:**
- WiFi Connection status
- API Endpoint URL
- Whether an API Key is set
- The device's unique ID
- The number of unsent data chunks in the backlog

## `WIFI_RESET`
Clears the stored WiFi credentials from non-volatile storage and restarts the device in Access Point mode ("Noctis-Setup"). This allows you to reconfigure the WiFi settings from a phone or computer.

**Usage:**
```
WIFI_RESET
```

## `WIFI_SCAN`
Scans for and lists available WiFi networks in the vicinity, showing their SSID, signal strength, and whether they are open or secured.

**Usage:**
```
WIFI_SCAN
```

## `SET_API [url]`
Sets the API endpoint URL where data chunks will be uploaded. This setting is saved to non-volatile storage.

**Usage:**
```
SET_API http://your-server-ip:5000/api/epochs:ingest-device
```

## `SET_API_KEY [key]`
Sets the Bearer token API key for authenticating with the backend. This setting is saved to non-volatile storage.

**Usage:**
```
SET_API_KEY your_secret_api_key_here
```

## `UPLOAD_NOW`
Manually triggers the process of uploading all unsent data chunks from the flash memory backlog. The device will attempt to upload them sequentially until the backlog is clear or an upload fails.

**Usage:**
```
UPLOAD_NOW
```
