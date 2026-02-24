#include "rgb_led.h"

// NOTE: Using GPIOs 12, 13, 14 for RGB LED. Please verify these are available on your hardware.
#define LED_R_PIN 14
#define LED_G_PIN 12
#define LED_B_PIN 13

void setupRGB() {
  pinMode(LED_R_PIN, OUTPUT);
  pinMode(LED_G_PIN, OUTPUT);
  pinMode(LED_B_PIN, OUTPUT);
  setRGBColor(0, 0, 0);
}

void setRGBColor(uint8_t r, uint8_t g, uint8_t b) {
  digitalWrite(LED_R_PIN, r > 0 ? HIGH : LOW);
  digitalWrite(LED_G_PIN, g > 0 ? HIGH : LOW);
  digitalWrite(LED_B_PIN, b > 0 ? HIGH : LOW);
}

// Simple digital on/off for now. Can be replaced with analogWrite for brightness control if needed.

void led_error_solid() {
  setRGBColor(255, 0, 0); // Red
}

void led_wifi_connecting() {
  setRGBColor(255, 255, 0); // Yellow
}

void led_connected_blinks() {
  setRGBColor(0, 255, 0); // Green
  delay(100);
  setRGBColor(0, 0, 0);
  delay(100);
  setRGBColor(0, 255, 0);
  delay(100);
  setRGBColor(0, 0, 0);
}

void led_offline_mode() {
  // Dim blue (20% brightness) - for now, just digital on
  setRGBColor(0, 0, 50); // Effectively just blue on
}

void led_uploading() {
  // Blue slow pulse - for now, just solid blue
  setRGBColor(0, 0, 255);
}

void led_off() {
  setRGBColor(0, 0, 0);
}
