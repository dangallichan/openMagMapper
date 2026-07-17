#include "Adafruit_MLX90395.h"
#include <Arduino_LSM9DS1.h>

// #define OPTIMAL_OSR MLX90395_OSR_8         // optimal OSR: 8 {1, 2, 4, 8}
#define OPTIMAL_OSR MLX90395_OSR_8         // optimal OSR: 8 {1, 2, 4, 8}

#define OPTIMAL_RESOLUTION MLX90395_RES_19 // optimal resolution: 19 {16, 17, 18, 19}
#define OPTIMAL_GAIN_SELECTION 7           // optimal gainSel: 7 {0-15}

// unsigned long time;

float ax, ay, az, gx, gy, gz; // for accel and gyro data

float NPARS = 3; // number of parameters to print: 3 for mag only, 9 for mag+accel+gyro

Adafruit_MLX90395 sensor = Adafruit_MLX90395();
void setup(void)
{
  Serial.begin(115200); // ensure baud in the Serial monitor is set to the same value

  // Wait for serial on USB platforms
  while (!Serial) {
      delay(10);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");  

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println("Hz");

  Serial.println("Starting Adafruit MLX90395 Demo");
  
  // Hardware I2C mode, can pass in address & alt Wire
  if (! sensor.begin_I2C()) {
    Serial.println("No sensor found ... check your wiring?");
    while (1) { delay(10); }
  }
  
  Serial.print("Found a MLX90395 sensor with unique id 0x");
  Serial.print(sensor.uniqueID[0], HEX);
  Serial.print(sensor.uniqueID[1], HEX);
  Serial.println(sensor.uniqueID[2], HEX);

  // Set and print OSR
  sensor.setOSR(OPTIMAL_OSR);
  Serial.print("OSR set to: ");
  switch (sensor.getOSR()) {
    case MLX90395_OSR_1: Serial.println("1 x"); break;
    case MLX90395_OSR_2: Serial.println("2 x"); break;
    case MLX90395_OSR_4: Serial.println("4 x"); break;
    case MLX90395_OSR_8: Serial.println("8 x"); break;
  }
  
  // Set and print Resolution
  sensor.setResolution(OPTIMAL_RESOLUTION);
  Serial.print("Resolution: ");
  switch (sensor.getResolution()) {
    case MLX90395_RES_16: Serial.println("16b"); break;
    case MLX90395_RES_17: Serial.println("17b"); break;
    case MLX90395_RES_18: Serial.println("18b"); break;
    case MLX90395_RES_19: Serial.println("19b"); break;
  }
  
  // Set and print Gain Selection
  sensor.setGain(OPTIMAL_GAIN_SELECTION);
  Serial.print("Gain: "); Serial.println(sensor.getGain());
}

void loop(void) {
  /* Get a new sensor event, normalized to uTesla */
  sensors_event_t event; 
  sensor.getEvent(&event);
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
  }
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az);

  }
  /* Display the results (magnetic field is measured in uTesla) */
  // time = micros();

  if (NPARS == 9) {
    Serial.print(micros());
    Serial.print(","); Serial.print(event.magnetic.x);
    Serial.print(","); Serial.print(event.magnetic.y); 
    Serial.print(","); Serial.print(event.magnetic.z); 
    Serial.print(","); Serial.print(ax);
    Serial.print(","); Serial.print(ay); 
    Serial.print(","); Serial.print(az); 
    Serial.print(","); Serial.print(gx);
    Serial.print(","); Serial.print(gy); 
    Serial.println(","); Serial.println(gz); 

  } else if (NPARS == 3) {
  Serial.print(micros());
  Serial.print(","); Serial.print(event.magnetic.x);
  Serial.print(","); Serial.print(event.magnetic.y); 
  Serial.print(","); Serial.println(event.magnetic.z); 
  }

  delay(50); // Update rate: 20 Hz
}




