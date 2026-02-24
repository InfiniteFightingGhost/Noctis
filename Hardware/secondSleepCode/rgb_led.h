#ifndef RGB_LED_H
#define RGB_LED_H

#include <Arduino.h>

void setupRGB();
void setRGBColor(uint8_t r, uint8_t g, uint8_t b);

// Status color definitions
void led_error_solid();
void led_wifi_connecting();
void led_connected_blinks();
void led_offline_mode();
void led_uploading();
void led_off();

#endif // RGB_LED_H
