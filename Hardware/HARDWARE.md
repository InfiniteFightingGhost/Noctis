# Noctis Sleep Monitor - Hardware Documentation

## Overview

The Noctis sleep monitor is a custom-built device that tracks sleep quality through multiple sensors. It combines millimeter-wave radar for vital signs detection with accelerometer-based movement tracking to provide comprehensive sleep analysis.

---

## Hardware Components

### Core Components

| Component | Model | Function | Interface |
|-----------|-------|----------|-----------|
| **Microcontroller** | ESP32 DevKit | Main processing unit | - |
| **mmWave Radar** | DFRobot C1001 (60GHz) | Heart rate, respiration, presence detection | UART (Serial2) |
| **Accelerometer/Gyro** | MPU6050 | Movement and vibration detection | I2C |
| **Real-Time Clock** | DS3231 | Timestamping (requires CR2032 battery) | I2C |
| **Flash Memory** | W25Q64 (8MB) | Data storage for offline recording | SPI |
| **Status LED** | Standard LED | Visual feedback | GPIO |

### Power Requirements

- **ESP32:** 3.3V (internal regulator from USB 5V)
- **C1001 Radar:** 5V, ~200mA
- **MPU6050:** 3.3V, ~3.5mA
- **DS3231:** 3.3V, ~200µA (plus CR2032 backup battery)
- **W25Q64:** 3.3V, ~15mA active, <1µA standby

**Total Power:** ~250mA @ 5V during active recording

---

## Pin Configuration

### Current Pin Mapping (Soldered Board)

```cpp
// SPI Flash
#define PIN_SPI_SCK    18
#define PIN_SPI_MISO   5
#define PIN_SPI_MOSI   23
#define PIN_FLASH_CS   19

// I2C (MPU6050 + DS3231)
#define PIN_I2C_SDA    2
#define PIN_I2C_SCL    15

// C1001 Radar (UART)
#define PIN_C1001_RX   27  // ESP32 RX ← C1001 TX
#define PIN_C1001_TX   26  // ESP32 TX → C1001 RX

// Status LED
#define LED_STATUS_PIN 32
```

### I2C Addresses

- **MPU6050:** 0x69 (AD0 pulled high)
- **DS3231:** 0x68 (fixed)

---

## Wiring Diagram

### C1001 Radar Connection
```
C1001 Module          ESP32
────────────────      ─────────────
VCC (5V)       ──→    5V
GND            ──→    GND
TX             ──→    GPIO 27 (RX)
RX             ←──    GPIO 26 (TX)
```

### MPU6050 Connection
```
MPU6050               ESP32
────────────────      ─────────────
VCC            ──→    3.3V
GND            ──→    GND
SDA            ←→     GPIO 2
SCL            ──→    GPIO 15
AD0            ──→    3.3V (sets I2C address to 0x69)
```

### DS3231 Connection
```
DS3231                ESP32
────────────────      ─────────────
VCC            ──→    3.3V
GND            ──→    GND
SDA            ←→     GPIO 2
SCL            ──→    GPIO 15
Battery        ──→    CR2032 (optional but recommended)
```

### W25Q64 Flash Connection
```
W25Q64                ESP32
────────────────      ─────────────
VCC            ──→    3.3V
GND            ──→    GND
CLK            ──→    GPIO 18
MISO (DO)      ──→    GPIO 5
MOSI (DI)      ──→    GPIO 23
CS             ──→    GPIO 19
```

---

## Sensor Specifications

### C1001 mmWave Radar

**Technology:** 60GHz millimeter-wave radar with Doppler sensing

**Detection Capabilities:**
- Heart rate: 40-180 BPM (±5 BPM accuracy)
- Respiration rate: 8-30 breaths/min (±2 breaths/min accuracy)
- Presence detection: Yes/No
- Sleep state classification: None/Deep/Light/Awake
- Turnover counting: Detects body position changes
- Apnea detection: Breathing pause events

**Operating Modes:**
- Sleep Mode (current): 40° beam, optimized for lying person
- Fall Detection Mode: 100° beam, wider coverage

**Range:**
- Minimum: ~0.8 meters
- Optimal: 1.0-2.0 meters
- Maximum: ~3.0 meters

**Limitations:**
- Requires clear line-of-sight to chest
- Works best when person is lying still
- Cannot detect through thick materials (mattresses, walls, metal)
- Single-person detection only
- 6-10 second warmup/lock-on time

### MPU6050 Accelerometer/Gyroscope

**Accelerometer:**
- Range: ±2g (configured)
- Resolution: 16-bit
- Sample Rate: 10 Hz (100ms intervals)

**Gyroscope:**
- Range: ±250°/s
- Resolution: 16-bit
- Sample Rate: 10 Hz (100ms intervals)

**Detection Capabilities:**
- Linear movement (acceleration delta)
- Rotational movement (gyroscope magnitude)
- Gravity-compensated movement
- Combined RMS-smoothed movement score

**Thresholds (Tunable):**
```cpp
#define VIB_LARGE_THRESHOLD 0.8   // Large movement threshold
#define VIB_MINOR_THRESHOLD 0.15  // Minor movement threshold
```

---

## Physical Placement Considerations

### Current Challenge: Dual Sensor Optimization

The device faces a fundamental placement conflict:

**C1001 Radar Requirements:**
- Needs 1-1.5m distance from chest
- Requires clear line-of-sight
- Best placed on nightstand pointing horizontally at bed

**MPU6050 Requirements:**
- Needs physical contact with bed frame/mattress
- Detects vibrations transmitted through bed structure
- Best placed under mattress or attached to frame

### Recommended Placement Options

#### Option 1: Under Mattress (Recommended for Testing)
```
    [Mattress]
       /\
      /  \
     /____\
    | MPU |  ← Device under mattress at chest level
    |C1001|
```

**Pros:**
- Optimal for MPU6050 (direct bed vibration)
- Protected from damage
- Unobtrusive

**Cons:**
- Radar signal may be attenuated by mattress
- Requires testing with specific mattress type

**Compatibility:**
- ✅ Thin foam (<5cm), air mattresses
- ⚠️ Memory foam (10-15cm) - test required
- ❌ Spring/coil mattresses (metal blocks RF)

#### Option 2: Nightstand Mount
```
[Nightstand]     1.5m      [Bed]
     |            ←→         /\
     |                      /  \
  [Device] ─────→          /____\
  Radar only              Person
```

**Pros:**
- Optimal for C1001 radar (clear line-of-sight)
- Easy to access for charging/data extraction
- Proven to work for vital signs

**Cons:**
- Poor vibration detection (only device's own movement)
- Movement data becomes "device stability" metric

#### Option 3: Side-Mounted on Bed Frame
```
         [Mattress]
            /\
           /  \
[Device]→→  o   (person lying down)
 on frame  MPU detects frame vibration
           C1001 aims across bed
```

**Pros:**
- Compromise between radar range and vibration
- Radar at ~0.8-1.0m (minimum effective range)
- MPU still detects bed structure movement

**Cons:**
- Radar at minimum range (suboptimal)
- May need testing for signal quality

---

## Data Storage Architecture

### Flash Memory Layout

```
Address Range         Sector    Contents
─────────────────────────────────────────────────
0x0000 - 0x0FFF      0         Reserved/unused
0x1000 - 0x1FFF      1         Metadata (12 bytes)
0x2000 - 0x2133      2         Chunk 0 (308 bytes)
0x2134 - 0x2267      2         Chunk 1 (308 bytes)
...                  ...       ...
0x7FFFFF             2047      End of 8MB
```

### Metadata Structure (12 bytes @ 0x1000)
```cpp
struct FlashMetadata {
  uint32_t writeAddr;    // Next write address
  uint32_t totalChunks;  // Chunks saved so far
  uint32_t magic;        // 0xDEADBEEF = valid
};
```

### Data Chunk Structure (308 bytes)

Each chunk contains 20 epochs (10 minutes of data):

```cpp
struct DataChunk {
  ChunkHeader header;     // 8 bytes
  EpochQ15 epochs[20];    // 15 bytes × 20 = 300 bytes
};
```

**Chunk Header (8 bytes):**
```cpp
struct ChunkHeader {
  uint32_t timestamp;     // Unix epoch (start of chunk)
  uint16_t num_epochs;    // Always 20
  uint16_t crc16;         // Data integrity check
};
```

**Epoch Data (15 bytes per 30-second epoch):**
```cpp
struct EpochQ15 {
  uint8_t in_bed_pct;        // 0-100%
  uint8_t hr_mean;           // Heart rate mean (BPM)
  uint8_t hr_std;            // HR standard deviation
  int8_t  dhr;               // HR delta from prev epoch (-128 to +127)
  uint8_t rr_mean;           // Respiration rate mean
  uint8_t rr_std;            // RR standard deviation
  int8_t  drr;               // RR delta from prev epoch
  uint8_t large_move_pct;    // Large movement %
  uint8_t minor_move_pct;    // Minor movement %
  uint8_t turnovers_delta;   // New turnovers this epoch
  uint8_t apnea_delta;       // New apnea events this epoch
  uint8_t flags;             // Sleep state (0-3)
  uint8_t vib_move_pct;      // Total vibration movement
  uint8_t vib_resp_q;        // Vibration-based quality (0-100)
  uint8_t agree_flags;       // Data quality flags (bitfield)
};
```

### Storage Capacity

- **Total capacity:** 8,388,608 bytes (8MB)
- **Usable storage:** ~8MB - 8KB (metadata) = 8,380,416 bytes
- **Chunks per flash:** 8,380,416 / 308 = ~27,208 chunks
- **Recording time:** 27,208 × 10 min = ~189 days continuous

---

## Power Considerations

### Normal Operation
- **Average current:** ~250mA @ 5V
- **Power consumption:** ~1.25W
- **USB power:** Adequate (standard USB provides 500mA)

### Battery Operation (Future)
For portable operation, recommended battery specifications:
- **Voltage:** 5V (via boost converter) or 3.7V LiPo with voltage regulator
- **Capacity:** 2000mAh minimum for ~8 hours
- **Type:** 18650 Li-ion (2500-3500mAh) or LiPo pouch

**Expected runtime:**
- 2500mAh @ 5V = 12.5Wh
- At 1.25W draw = ~10 hours runtime

### Power Optimization (Not Implemented)
Potential improvements for battery operation:
- Deep sleep between epochs (~200mA → ~20mA)
- Flash power-down when not writing (~15mA → ~1µA)
- Radar duty cycling (not recommended - affects detection)

---

## Bill of Materials (BOM)

| Component | Quantity | Est. Cost (USD) | Source |
|-----------|----------|-----------------|--------|
| ESP32 DevKit | 1 | $8-12 | Amazon/AliExpress |
| DFRobot C1001 Radar | 1 | $25-35 | DFRobot/Mouser |
| MPU6050 Module | 1 | $2-5 | Amazon/AliExpress |
| DS3231 RTC Module | 1 | $3-6 | Amazon/AliExpress |
| W25Q64 Flash (8MB) | 1 | $1-3 | AliExpress/LCSC |
| CR2032 Battery | 1 | $1-2 | Local store |
| LED + 330Ω resistor | 1 | $0.50 | - |
| Breadboard/PCB | 1 | $3-10 | - |
| USB Cable (Micro/Type-C) | 1 | $2-5 | - |
| Jumper wires | ~15 | $2-5 | - |
| **Total** | - | **~$50-85** | - |

*Prices as of 2024. Bulk purchasing can reduce costs significantly.*

---

## Assembly Notes

### Critical Wiring Considerations

1. **C1001 Power:** Must be 5V, not 3.3V. Drawing ~200mA, use direct USB 5V line.

2. **I2C Pull-ups:** Most breakout boards have built-in pull-ups. If using raw chips, add 4.7kΩ resistors from SDA/SCL to 3.3V.

3. **MPU6050 AD0 Pin:** 
   - Pull high (3.3V) for address 0x69
   - Leave floating/ground for address 0x68
   - Firmware configured for 0x69

4. **SPI Shared Bus:** 
   - MISO, MOSI, SCK shared between flash and any future SPI devices
   - Each device needs unique CS (chip select) pin

5. **RTC Battery:**
   - CR2032 optional but highly recommended
   - Without it, time resets to 2000-01-01 on every power loss
   - Use `SET_TIME` command after boot if no battery

### Soldering Best Practices

- Use flux for clean solder joints
- Test continuity with multimeter before powering
- Double-check TX/RX connections (commonly swapped)
- Verify 5V and 3.3V rails before connecting components
- Add heat shrink or hot glue for strain relief on wires

### Enclosure Requirements

If designing a 3D-printed or fabricated enclosure:

1. **C1001 Radar Face:** Must be unobstructed (no plastic >2mm in front)
2. **USB Port:** Easy access for charging and data extraction
3. **LED Visibility:** Status LED visible from outside
4. **Ventilation:** Small vent holes for heat dissipation
5. **Mounting:** Mechanism to secure to nightstand or bed frame

---

## Troubleshooting Hardware Issues

### C1001 Radar Not Responding

**Symptoms:** `HR=0 RR=0 Presence=0` constantly

**Checks:**
1. Verify 5V power to radar (measure with multimeter)
2. Check TX/RX not swapped (common issue)
3. Test with diagnostic sketch to verify communication
4. Ensure radar LED is blinking (indicates power and operation)
5. Try different baud rates (115200, 9600, 256000)

**Solution:** Most commonly TX/RX swapped or insufficient 5V current

### MPU6050 Not Detected

**Symptoms:** `MPU6050 Fail!` at boot, error code 2 (LED blinks twice)

**Checks:**
1. Verify I2C wiring (SDA ↔ GPIO 2, SCL → GPIO 15)
2. Check 3.3V power to MPU
3. Verify AD0 pin pulled high (for address 0x69)
4. Use I2C scanner sketch to detect address

**Solution:** Check I2C address matches firmware (0x69)

### DS3231 Time Resets

**Symptoms:** Time shows year 2000 after power cycle

**Checks:**
1. Check if CR2032 battery installed
2. Measure battery voltage (should be >2.5V)
3. Verify I2C communication working

**Solution:** Install fresh CR2032 battery, or use `SET_TIME` command after every boot

### Flash Write Failures

**Symptoms:** `FLASH WRITE FAILED` messages, or `totalChunks=0` when extracting

**Checks:**
1. Verify all SPI pins connected correctly
2. Check 3.3V power stable
3. Verify CS pin (GPIO 19) not conflicting with other uses
4. Run flash diagnostic test

**Solution:** 
- Ensure sector erased before writing (firmware v3.5+ handles this)
- Try slower SPI frequency: `SPI.setFrequency(500000);`

### No Vibration Data

**Symptoms:** `large_move_pct=0%, minor_move_pct=0%` even when moving

**Checks:**
1. MPU6050 connected and initialized
2. Thresholds may be too high - watch Serial Monitor "VIB rms=" values
3. Device placement (nightstand vs bed affects sensitivity)

**Solution:** 
- Lower thresholds: `VIB_LARGE_THRESHOLD = 0.3`, `VIB_MINOR_THRESHOLD = 0.05`
- Place device on/under bed for better vibration coupling

---

## Hardware Limitations & Future Improvements

### Current Limitations

1. **Single-person detection:** C1001 cannot distinguish multiple people
2. **Position-dependent:** Radar requires specific placement for HR/RR
3. **Mattress interference:** Thick/metallic mattresses may block radar
4. **Wired power only:** No battery option currently implemented
5. **Manual data extraction:** Requires PC connection, no wireless upload

### Potential Hardware Upgrades

1. **WiFi Data Upload:**
   - ESP32 already has WiFi capability
   - Add firmware to upload chunks to cloud/server
   - Enable real-time monitoring

2. **Battery Operation:**
   - Add 18650 Li-ion battery + TP4056 charging module
   - Implement deep sleep for power saving
   - Target 24h+ runtime per charge

3. **Multiple Sensors:**
   - Add temperature/humidity sensor (BME280)
   - Add ambient light sensor (for bedroom darkness tracking)
   - Add sound level sensor (snoring detection)

4. **Improved Radar:**
   - DFRobot offers newer models with better penetration
   - Consider dual-radar setup for multiple people

5. **Custom PCB:**
   - Reduce size by 70%
   - Improve reliability vs breadboard/wire connections
   - Enable mass production

---

## Safety & Compliance

### RF Exposure (C1001 Radar)

- **Frequency:** 60 GHz (EHF band)
- **Power:** <10 mW (low power device)
- **Safety:** Non-ionizing radiation, well below FCC/CE limits
- **Distance:** Designed for >0.8m operation (additional safety margin)

The 60GHz frequency used is in the ISM (Industrial, Scientific, Medical) band and is considered safe for continuous exposure at the device's power levels.

### Electrical Safety

- **Operating voltage:** 5V DC (USB)
- **Low voltage device:** No electrical hazard to users
- **Battery safety:** If adding Li-ion battery, use protection circuit

### Regulatory Compliance (Future)

For commercial production, device would require:
- **FCC Part 15** (USA) for RF emissions
- **CE marking** (Europe) for electromagnetic compatibility
- **RoHS compliance** for lead-free manufacturing

---

## Maintenance

### Regular Maintenance

1. **Battery replacement:** CR2032 in DS3231 every 2-3 years
2. **Cleaning:** Wipe radar face with soft cloth (no obstructions)
3. **Firmware updates:** Check GitHub for bug fixes and improvements

### Flash Memory Longevity

- **Write endurance:** ~100,000 cycles per sector
- **With current usage:** Sector 1 (metadata) rewritten every 10 minutes
- **Expected lifespan:** 100,000 × 10 min = ~1.9 years before sector 1 degrades
- **Mitigation:** Future firmware could rotate metadata across multiple sectors

### Data Backup

Flash memory can fail. Recommended practices:
- Extract and backup data weekly
- Store CSV files in cloud storage
- Consider adding SD card for redundant storage

---

## Version History

### Hardware Revision 1.0 (Current)
- Initial breadboard/soldered prototype
- All core components functional
- Placement optimization ongoing

### Planned Hardware Revision 2.0
- Custom PCB design
- Battery operation
- Smaller form factor
- Improved RF antenna design for mattress penetration

---

## Additional Resources

### Datasheets
- [ESP32 Technical Reference](https://www.espressif.com/sites/default/files/documentation/esp32_technical_reference_manual_en.pdf)
- [DFRobot C1001 Wiki](https://wiki.dfrobot.com/SKU_SEN0623_C1001_mmWave_Human_Detection_Sensor)
- [MPU6050 Datasheet](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf)
- [DS3231 Datasheet](https://datasheets.maximintegrated.com/en/ds/DS3231.pdf)
- [W25Q64 Datasheet](https://www.winbond.com/resource-files/w25q64jv%20revj%2003272018%20plus.pdf)

### Arduino Libraries Used
- `Wire` (I2C) - Built-in
- `SPI` - Built-in
- `RTClib` - Adafruit
- `SPIMemory` - Marzogh
- `Adafruit_MPU6050` - Adafruit
- `Adafruit_Sensor` - Adafruit
- `DFRobot_HumanDetection` - DFRobot
- `esp_task_wdt` - ESP32 built-in

### Community & Support
- GitHub Issues: [Report hardware problems]
- DFRobot Forum: [C1001-specific questions]
- ESP32 Community: [General ESP32 help]

---

## License

Hardware design is open source under [specify license - MIT/CC-BY-SA/etc].

Component datasheets and libraries retain their original licenses.

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Maintainer:** [Your name/team]  
**Status:** Active Development
