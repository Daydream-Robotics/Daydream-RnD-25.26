#include <Arduino.h>
#include <Adafruit_BNO08x.h>
#include <SPI.h>

// GOAL: get yaw

/* * WIRING FOR NANO 33 IOT (SPI)
 * ----------------------------
 * BNO08x VIN  -> 3.3V
 * BNO08x GND  -> GND
 * BNO08x SCK  -> D13
 * BNO08x SDA  -> D12 (MISO)
 * BNO08x DI   -> D11 (MOSI)
 * BNO08x CS   -> D10
 * BNO08x INT  -> D9
 * BNO08x RST  -> D8
 * BNO08x PS0  -> 3.3V (Crucial for SPI mode)
 * BNO08x PS1  -> 3.3V (Crucial for SPI mode)
 */

// SPI Pins
#define BNO08X_CS 10
#define BNO08X_INT 9
#define BNO08X_RESET 8

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Using GAME_ROTATION_VECTOR (No Magnetometer) for faster, stable relative yaw
sh2_SensorId_t reportType = SH2_GAME_ROTATION_VECTOR;

void setReports();
double getYaw();

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  Serial.println("Orientation Sensor Test (SPI Mode)"); Serial.println("");

  // Initialize SPI
  // The library automatically uses the default SPI pins for Nano 33 IoT (SCK=13, MISO=12, MOSI=11)
  if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT)) {
    Serial.println("Failed to find BNO08x chip via SPI!");
    Serial.println("CHECK: Are PS0 and PS1 connected to 3.3V?");
    while (1) { delay(10); }
  }
  
  Serial.println("BNO08x Found over SPI");

  // Debug: Print Product IDs to confirm good communication
  for (int n = 0; n < bno08x.prodIds.numEntries; n++) {
    Serial.print("Part ");
    Serial.print(bno08x.prodIds.entry[n].swPartNumber);
    Serial.print(": Version :");
    Serial.print(bno08x.prodIds.entry[n].swVersionMajor);
    Serial.print(".");
    Serial.print(bno08x.prodIds.entry[n].swVersionMinor);
    Serial.print(".");
    Serial.print(bno08x.prodIds.entry[n].swVersionPatch);
    Serial.print(" Build ");
    Serial.println(bno08x.prodIds.entry[n].swBuildNumber);
  }

  setReports();

  Serial.println("BNO08x initialized");
  delay(100);
}

void loop() {
  if (bno08x.wasReset()) {
    Serial.print("Sensor was reset ");
    setReports();
  }

  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_GAME_ROTATION_VECTOR) {
      double yaw = getYaw();
      
      // Convert Radians to Degrees for easier reading
      Serial.print("Yaw: ");
      Serial.println(yaw * (180.0 / PI));
    }
  }
}

double getYaw() {
  double i = sensorValue.un.gameRotationVector.i;
  double j = sensorValue.un.gameRotationVector.j;
  double k = sensorValue.un.gameRotationVector.k;
  double real = sensorValue.un.gameRotationVector.real;

  // Calculate Yaw (rotation around Z axis)
  // This formula converts Quaternion to Yaw in radians
  // Note: This returns data in range -PI to +PI (-180 to +180)
  return atan2(2.0 * (real * k + i * j), 1.0 - 2.0 * (j * j + k * k));
}

void setReports() {
  Serial.println("Setting desired reports");
  // 5000 microseconds = 5ms = 200Hz update rate
  if (! bno08x.enableReport(reportType, 5000)) {
    Serial.println("Could not enable game vector");
  }
}