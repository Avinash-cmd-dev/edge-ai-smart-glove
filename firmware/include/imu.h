#ifndef IMU_H
#define IMU_H

struct IMUSample
{
    float ax, ay, az; // g
    float gx, gy, gz; // deg/s
};

class IMU
{
public:
    bool begin();
    bool readSample(IMUSample &sample);

private:
    static constexpr uint8_t I2C_ADDRESS = 0x68;
    static constexpr float ACCEL_SENSITIVITY = 16384.0f; // LSB/g at +/-2g
    static constexpr float GYRO_SENSITIVITY = 131.0f;    // LSB/(deg/s) at +/-250dps

    void writeRegister(uint8_t reg, uint8_t value);
};

#endif
