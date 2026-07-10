#include <Arduino.h>
#include <Wire.h>
#include "imu.h"

namespace
{
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;
constexpr uint8_t REG_WHO_AM_I = 0x75;
constexpr uint8_t EXPECTED_WHO_AM_I = 0x68;
} // namespace

void IMU::writeRegister(uint8_t reg, uint8_t value)
{
    Wire.beginTransmission(I2C_ADDRESS);
    Wire.write(reg);
    Wire.write(value);
    Wire.endTransmission();
}

bool IMU::begin()
{
    Wire.begin();

    // Wake the MPU6050 up (it starts in sleep mode).
    writeRegister(REG_PWR_MGMT_1, 0x00);
    delay(50);

    // +/-2g accelerometer range, +/-250 deg/s gyroscope range.
    writeRegister(REG_ACCEL_CONFIG, 0x00);
    writeRegister(REG_GYRO_CONFIG, 0x00);

    Wire.beginTransmission(I2C_ADDRESS);
    Wire.write(REG_WHO_AM_I);
    if (Wire.endTransmission(false) != 0)
    {
        return false;
    }

    if (Wire.requestFrom((uint8_t)I2C_ADDRESS, (uint8_t)1) != 1)
    {
        return false;
    }

    return Wire.read() == EXPECTED_WHO_AM_I;
}

bool IMU::readSample(IMUSample &sample)
{
    Wire.beginTransmission(I2C_ADDRESS);
    Wire.write(REG_ACCEL_XOUT_H);
    if (Wire.endTransmission(false) != 0)
    {
        return false;
    }

    const uint8_t bytesRequested = 14; // accel (6) + temp (2) + gyro (6)
    if (Wire.requestFrom((uint8_t)I2C_ADDRESS, bytesRequested) != bytesRequested)
    {
        return false;
    }

    int16_t rawAx = (Wire.read() << 8) | Wire.read();
    int16_t rawAy = (Wire.read() << 8) | Wire.read();
    int16_t rawAz = (Wire.read() << 8) | Wire.read();
    Wire.read();
    Wire.read(); // discard temperature
    int16_t rawGx = (Wire.read() << 8) | Wire.read();
    int16_t rawGy = (Wire.read() << 8) | Wire.read();
    int16_t rawGz = (Wire.read() << 8) | Wire.read();

    sample.ax = rawAx / ACCEL_SENSITIVITY;
    sample.ay = rawAy / ACCEL_SENSITIVITY;
    sample.az = rawAz / ACCEL_SENSITIVITY;
    sample.gx = rawGx / GYRO_SENSITIVITY;
    sample.gy = rawGy / GYRO_SENSITIVITY;
    sample.gz = rawGz / GYRO_SENSITIVITY;

    return true;
}
