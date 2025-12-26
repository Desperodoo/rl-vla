import serial
import time
import os

# 优先使用固定的符号链接，如果不存在则回退到 ttyUSB0
if os.path.exists('/dev/uarm_servo'):
    SERIAL_PORT = '/dev/uarm_servo'
else:
    SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200

print(f"Using serial port: {SERIAL_PORT}")
with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5) as ser:
    # Modify ID: change original ID=000 to 000
    cmd = b'#000PID000!\r\n'
    ser.write(cmd)
    print(f"Sent ID modification command: {cmd.decode().strip()}")
    time.sleep(0.5)

    if ser.in_waiting:
        response = ser.read(ser.in_waiting).decode(errors='ignore')
        print(f"Response: {response}")
    else:
        print("No servo response received (modification may have succeeded but no feedback)")
