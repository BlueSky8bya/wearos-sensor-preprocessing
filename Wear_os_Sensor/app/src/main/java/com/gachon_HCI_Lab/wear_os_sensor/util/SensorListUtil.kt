package com.example.wear_os_sensor

import android.hardware.Sensor
import com.samsung.android.service.health.tracking.data.HealthTrackerType

class SensorListUtil {
    private val sensorName: HashMap<Any, String> = hashMapOf(
        Sensor.TYPE_ACCELEROMETER to "Accelerometer",
        Sensor.TYPE_GYROSCOPE to "Gyroscope",
        Sensor.TYPE_GRAVITY to "Gravity",
//        2 to "Magnetometer",
//        15 to "Game Rotation" ,
        Sensor.TYPE_LIGHT to "Light" ,
//        6 to "Barometer",
        Sensor.TYPE_STEP_DETECTOR to "Step counter",
        Sensor.TYPE_HEART_RATE to "Heart Rate",
        HealthTrackerType.PPG_GREEN to "ppg Green",
//        HealthTrackerType.PPG_RED to "ppg Red",
//        HealthTrackerType.PPG_IR to "ppg Ir"
        "other sensors" to "other sensors"
    )

    fun getSensorName(num: Any): String? {
        return sensorName[num]
    }

    fun getSensorIDList(): MutableList<Any> {
        return sensorName.keys.toMutableList()
    }

}