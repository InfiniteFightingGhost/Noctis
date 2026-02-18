#!/usr/bin/env python3
"""
ESP32 Sleep Monitor - Flash Data Extractor
Reads binary chunk data from ESP32 and converts to CSV
"""

import serial
import struct
import sys
import time
import pandas as pd
from datetime import datetime

# Data structure sizes
EPOCH_SIZE = 15  # bytes
CHUNK_HEADER_SIZE = 8  # bytes
EPOCHS_PER_CHUNK = 20
CHUNK_SIZE = CHUNK_HEADER_SIZE + (EPOCH_SIZE * EPOCHS_PER_CHUNK)  # 308 bytes

def extract_sleep_data(port, output_csv='sleep_data.csv'):
    """
    Extract sleep data from ESP32 flash via serial
    
    Args:
        port: Serial port (e.g., 'COM5' or '/dev/ttyUSB0')
        output_csv: Output CSV filename
    
    Returns:
        True if successful, False otherwise
    """
    print("╔════════════════════════════════════════════════════╗")
    print("║   Sleep Data Extractor v1.0                        ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    # Open serial connection
    print(f"Connecting to {port}...")
    try:
        ser = serial.Serial(port, 115200, timeout=5)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False
    
    time.sleep(0.5)
    
    # Reset ESP32 to get fresh boot
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║  >>> PRESS THE RESET BUTTON ON YOUR ESP32 NOW <<< ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    
    # Wait for ESP32 to boot (look for ready prompt)
    print("Waiting for ESP32 to boot...")
    boot_timeout = 10
    start = time.time()
    boot_complete = False
    
    while time.time() - start < boot_timeout:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"  {line}")
            if "Commands:" in line or "DUMP" in line:
                boot_complete = True
                break
    
    if not boot_complete:
        print("⚠ Did not see boot complete message, trying anyway...")
    else:
        print("✓ ESP32 is ready!")
    
    time.sleep(1)
    
    # Send DUMP command
    print()
    print("Sending DUMP command...")
    ser.write(b'DUMP\n')
    ser.flush()
    time.sleep(0.5)
    
    # Wait for acknowledgment
    ack_received = False
    for _ in range(50):
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  {line}")
            if 'DUMP_START_ACK' in line:
                ack_received = True
                print("✓ ESP32 acknowledged download request")
                break
        time.sleep(0.1)
    
    if not ack_received:
        print("✗ No acknowledgment from ESP32")
        ser.close()
        return False
    
    # Read metadata
    total_chunks = 0
    chunk_size = CHUNK_SIZE
    
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'TOTAL_CHUNKS:' in line:
                total_chunks = int(line.split(':')[1])
                print(f"Total chunks to download: {total_chunks}")
            elif 'CHUNK_SIZE:' in line:
                chunk_size = int(line.split(':')[1])
                print(f"Chunk size: {chunk_size} bytes")
            elif 'DATA_BEGIN' in line:
                print()
                print("=" * 50)
                print("Starting data download...")
                print("=" * 50)
                print()
                break
        time.sleep(0.05)
    
    if total_chunks == 0:
        print("⚠ No chunks to download (totalChunks = 0)")
        print()
        print("Possible causes:")
        print("  1. Recording firmware hasn't saved any data yet")
        print("  2. Wrong extractor firmware (check metadata address)")
        print("  3. Flash was erased")
        ser.close()
        return False
    
    # Read chunks
    chunks_data = []
    chunk_counter = 0
    
    while chunk_counter < total_chunks:
        # Wait for chunk marker
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if 'CHUNK:' in line:
                chunk_num = int(line.split(':')[1])
                
                # Read binary chunk data
                chunk_bytes = ser.read(chunk_size)
                
                if len(chunk_bytes) == chunk_size:
                    chunks_data.append(chunk_bytes)
                    chunk_counter += 1
                    
                    # Progress indicator
                    progress = (chunk_counter / total_chunks) * 100
                    print(f"Progress: {chunk_counter}/{total_chunks} ({progress:.1f}%)")
                else:
                    print(f"⚠ Warning: Chunk {chunk_num} incomplete ({len(chunk_bytes)} bytes)")
                
                # Read trailing newline
                ser.readline()
            
            elif 'DATA_END' in line:
                print()
                print("Download complete!")
                break
    
    ser.close()
    
    # Parse chunks into epochs
    print()
    print("Parsing data...")
    
    all_epochs = []
    
    for chunk_idx, chunk_bytes in enumerate(chunks_data):
        # Parse chunk header (8 bytes)
        timestamp, num_epochs, crc16 = struct.unpack('<IHH', chunk_bytes[:8])
        
        # Parse epochs (15 bytes each)
        for epoch_idx in range(num_epochs):
            offset = 8 + (epoch_idx * 15)
            epoch_bytes = chunk_bytes[offset:offset+15]
            
            # Unpack epoch: 9 unsigned bytes, 2 signed bytes, 4 unsigned bytes
            values = struct.unpack('<9B2b4B', epoch_bytes)
            
            # Calculate epoch timestamp (30 seconds per epoch)
            epoch_timestamp = timestamp + (epoch_idx * 30)
            epoch_datetime = datetime.fromtimestamp(epoch_timestamp)
            
            epoch_data = {
                'datetime': epoch_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': epoch_timestamp,
                'in_bed_pct': values[0],
                'hr_mean': values[1],
                'hr_std': values[2],
                'dhr': values[3],  # signed
                'rr_mean': values[4],
                'rr_std': values[5],
                'drr': values[6],  # signed
                'large_move_pct': values[7],
                'minor_move_pct': values[8],
                'turnovers_delta': values[9],
                'apnea_delta': values[10],
                'flags': values[11],
                'vib_move_pct': values[12],
                'vib_resp_q': values[13],
                'agree_flags': values[14]
            }
            
            all_epochs.append(epoch_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(all_epochs)
    df.to_csv(output_csv, index=False)
    
    print(f"✓ Extracted {len(all_epochs)} epochs")
    print(f"✓ Saved to: {output_csv}")
    print()
    
    # Summary stats
    print("=== Sleep Summary ===")
    print(f"Recording duration: {len(all_epochs) * 30 / 3600:.1f} hours")
    print(f"Start time: {df['datetime'].iloc[0]}")
    print(f"End time:   {df['datetime'].iloc[-1]}")
    print(f"Average HR: {df['hr_mean'][df['hr_mean'] > 0].mean():.1f} BPM")
    print(f"Average RR: {df['rr_mean'][df['rr_mean'] > 0].mean():.1f} breaths/min")
    print(f"In bed %:   {df['in_bed_pct'].mean():.1f}%")
    print(f"Total turnovers: {df['turnovers_delta'].sum()}")
    print(f"Total apnea events: {df['apnea_delta'].sum()}")
    print()
    
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_flash_data.py <serial_port> [output.csv]")
        print()
        print("Examples:")
        print("  python extract_flash_data.py COM5")
        print("  python extract_flash_data.py /dev/ttyUSB0 my_sleep_data.csv")
        sys.exit(1)
    
    port = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'sleep_data.csv'
    
    success = extract_sleep_data(port, output)
    
    if success:
        print("✓✓✓ Extraction successful! ✓✓✓")
        sys.exit(0)
    else:
        print("✗✗✗ Extraction failed ✗✗✗")
        sys.exit(1)
