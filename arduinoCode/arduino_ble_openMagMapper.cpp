// This code generated purely by AI and hasn't been tested yet

#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <Adafruit_MLX90395.h>
#include <Wire.h>

// BLE Service and Characteristics (custom 128-bit UUIDs)
const char* BLE_SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char* BLE_ACCEL_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214";
const char* BLE_GYRO_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214";
const char* BLE_MAG_UUID = "19B10003-E8F2-537E-4F6C-D104768A1214";
const char* BLE_MAGMLX_UUID = "19B10004-E8F2-537E-4F6C-D104768A1214";

BLEService sensorService(BLE_SERVICE_UUID);
BLECharacteristic imuAccelChar(BLE_ACCEL_UUID, BLERead | BLENotify, 12);
BLECharacteristic imuGyroChar(BLE_GYRO_UUID, BLERead | BLENotify, 12);
BLECharacteristic imuMagChar(BLE_MAG_UUID, BLERead | BLENotify, 12);
BLECharacteristic mlxMagChar(BLE_MAGMLX_UUID, BLERead | BLENotify, 12);


#define OPTIMAL_OSR MLX90395_OSR_8         // optimal OSR: 8 {1, 2, 4, 8}
#define OPTIMAL_RESOLUTION MLX90395_RES_19 // optimal resolution: 19 {16, 17, 18, 19}
#define OPTIMAL_GAIN_SELECTION 7           // optimal gainSel: 7 {0-15}


Adafruit_MLX90395 mlx;
bool mlxAvailable = false;
unsigned long lastMlxReadWarnMs = 0;
unsigned long ledCycleStartMs = 0;
bool ledPulseActive = false;

const unsigned long LED_STREAM_PULSE_PERIOD_MS = 1200;
const unsigned long LED_STREAM_PULSE_OFF_MS = 150;

// Buffer for sensor data
uint8_t imuAccelData[12];
uint8_t imuGyroData[12];
uint8_t imuMagData[12];
uint8_t mlxMagData[12];

void updateServingLed(bool servingMagmlx) {
  if (!servingMagmlx) {
    digitalWrite(LED_BUILTIN, HIGH);
    ledPulseActive = false;
    return;
  }

  unsigned long nowMs = millis();
  if (!ledPulseActive) {
    if (nowMs - ledCycleStartMs >= LED_STREAM_PULSE_PERIOD_MS) {
      ledPulseActive = true;
      ledCycleStartMs = nowMs;
      digitalWrite(LED_BUILTIN, LOW);
    }
    return;
  }

  if (nowMs - ledCycleStartMs >= LED_STREAM_PULSE_OFF_MS) {
    digitalWrite(LED_BUILTIN, HIGH);
    ledPulseActive = false;
  }
}

void setup() {
  Serial.begin(115200);
  // Do not block on the USB serial port; BLE should start even when no host
  // terminal is attached (for example when powered from a battery pack).

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  // Initialize MLX90395 (optional)
  mlxAvailable = initMLX90395();
  if (!mlxAvailable) {
    Serial.println("Warning: MLX90395 not detected, continuing without magnetometer data.");
  }
  
  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE!");
    while (1);
  }
  
  // Set up BLE
  BLE.setLocalName("Nano33BLE_Sensor");
  BLE.setAdvertisedService(sensorService);
  
  sensorService.addCharacteristic(imuAccelChar);
  sensorService.addCharacteristic(imuGyroChar);
  sensorService.addCharacteristic(imuMagChar);
  sensorService.addCharacteristic(mlxMagChar);
  
  BLE.addService(sensorService);

  Serial.println("BLE UUID config:");
  Serial.print("  Service: ");
  Serial.println(BLE_SERVICE_UUID);
  Serial.print("  Accel:   ");
  Serial.println(BLE_ACCEL_UUID);
  Serial.print("  Gyro:    ");
  Serial.println(BLE_GYRO_UUID);
  Serial.print("  Mag:     ");
  Serial.println(BLE_MAG_UUID);
  Serial.print("  MagMLX:  ");
  Serial.println(BLE_MAGMLX_UUID);
  
  // Start advertising
  BLE.advertise();
  
  Serial.println("BLE Sensor broadcasting started");
}

void loop() {
  BLEDevice central = BLE.central();
  
  if (central) {
    Serial.print("Connected to: ");
    Serial.println(central.address());
    
    while (central.connected()) {
      bool servingMagmlx = false;

      // Read IMU data
      float accelX, accelY, accelZ;
      float gyroX, gyroY, gyroZ;
      float magX, magY, magZ;
      
      if (IMU.accelerationAvailable()) {
        IMU.readAcceleration(accelX, accelY, accelZ);
        packFloatData(imuAccelData, accelX, accelY, accelZ);
        imuAccelChar.writeValue(imuAccelData, 12);
      }
      
      if (IMU.gyroscopeAvailable()) {
        IMU.readGyroscope(gyroX, gyroY, gyroZ);
        packFloatData(imuGyroData, gyroX, gyroY, gyroZ);
        imuGyroChar.writeValue(imuGyroData, 12);
      }

      if (IMU.magneticFieldAvailable()) {
        IMU.readMagneticField(magX, magY, magZ);
        packFloatData(imuMagData, magX, magY, magZ);
        imuMagChar.writeValue(imuMagData, 12);
      }
      
      // Read MLX90395 magnetometer data when available.
      if (mlxAvailable) {
        int16_t mlxMagX, mlxMagY, mlxMagZ;
        if (readMLX90395(mlxMagX, mlxMagY, mlxMagZ)) {
          packInt16Data(mlxMagData, mlxMagX, mlxMagY, mlxMagZ);
          mlxMagChar.writeValue(mlxMagData, 12);
          servingMagmlx = true;
        } else {
          // Throttle warning output to avoid spamming serial logs.
          unsigned long nowMs = millis();
          if (nowMs - lastMlxReadWarnMs >= 1000) {
            Serial.println("Warning: MLX90395 read failed.");
            lastMlxReadWarnMs = nowMs;
          }
        }
      }

      updateServingLed(servingMagmlx);
      
      delay(50); // Update rate: 20 Hz
    }

    digitalWrite(LED_BUILTIN, HIGH);
    
    Serial.println("Disconnected");
  }
}

bool initMLX90395() {
  if (!mlx.begin_I2C()) {
    return false;
  }

  mlx.setOSR(OPTIMAL_OSR);
  mlx.setGain(OPTIMAL_GAIN_SELECTION);
  mlx.setResolution(OPTIMAL_RESOLUTION);
  
  return true;
}

bool readMLX90395(int16_t &x, int16_t &y, int16_t &z) {
  float mx, my, mz;
  if (!mlx.readData(&mx, &my, &mz)) {
    return false;
  }

  x = (int16_t)mx;
  y = (int16_t)my;
  z = (int16_t)mz;

  return true;
}

void packFloatData(uint8_t *buffer, float x, float y, float z) {
  memcpy(buffer, &x, 4);
  memcpy(buffer + 4, &y, 4);
  memcpy(buffer + 8, &z, 4);
}

void packInt16Data(uint8_t *buffer, int16_t x, int16_t y, int16_t z) {
  buffer[0] = (x >> 8) & 0xFF;
  buffer[1] = x & 0xFF;
  buffer[2] = (y >> 8) & 0xFF;
  buffer[3] = y & 0xFF;
  buffer[4] = (z >> 8) & 0xFF;
  buffer[5] = z & 0xFF;
  buffer[6] = 0;
  buffer[7] = 0;
  buffer[8] = 0;
  buffer[9] = 0;
  buffer[10] = 0;
  buffer[11] = 0;
}
