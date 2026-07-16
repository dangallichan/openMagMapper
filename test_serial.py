import serial
import time

ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # 等待Arduino重启完成

for _ in range(30):
    line = ser.readline()
    print(line)

ser.close()