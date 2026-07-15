// This code generated purely by AI and hasn't been tested yet

#include <ArduinoBLE.h>
#include <Arduino_LSM6DSOX.h>
#include <Adafruit_MLX90395.h>
#include <Wire.h>

// BLE Service and Characteristics
BLEService sensorService("180A");
BLECharacteristic imuAccelChar("2A58", BLERead | BLENotify, 12);
BLECharacteristic imuGyroChar("2A59", BLERead | BLENotify, 12);
BLECharacteristic magnetometerChar("2A5B", BLERead | BLENotify, 12);

Adafruit_MLX90395 mlx;

// Buffer for sensor data
uint8_t imuAccelData[12];
uint8_t imuGyroData[12];
uint8_t magnetometerData[12];

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  
  // Initialize MLX90395
  if (!initMLX90395()) {
    Serial.println("Failed to initialize MLX90395!");
    while (1);
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
  sensorService.addCharacteristic(magnetometerChar);
  
  BLE.addService(sensorService);
  
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
      // Read IMU data
      float accelX, accelY, accelZ;
      float gyroX, gyroY, gyroZ;
      
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
      
      // Read MLX90395 magnetometer data
      int16_t magX, magY, magZ;
      if (readMLX90395(magX, magY, magZ)) {
        packInt16Data(magnetometerData, magX, magY, magZ);
        magnetometerChar.writeValue(magnetometerData, 12);
      }
      
      delay(100); // Update rate: 10 Hz
    }
    
    Serial.println("Disconnected");
  }
}

bool initMLX90395() {
  if (!mlx.begin_I2C()) {
    return false;
  }

  mlx.setOSR(MLX90395_OSR_3);
  mlx.setFilter(MLX90395_FILTER_5);
  mlx.setResolution(MLX90395_X, MLX90395_RES_19);
  mlx.setResolution(MLX90395_Y, MLX90395_RES_19);
  mlx.setResolution(MLX90395_Z, MLX90395_RES_16);

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
