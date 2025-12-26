package com.example.wear_os_sensor

import android.hardware.Sensor
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log

class SensorManagerUtil {
    private val sensorManager: SensorManager
    constructor(sensorManager: SensorManager){
        this.sensorManager = sensorManager
    }

    fun registerSensor(listener: SensorEventListener, list: ArrayList<Int>): ArrayList<Int>{
        var sensorList: ArrayList<Int> = arrayListOf()
        for (sensorType: Int in list){
            //안드로이드에서 api를 열어둔 센서
            if (sensorType >= Sensor.TYPE_DEVICE_PRIVATE_BASE) continue
            //장비가 센서를 지원하는지 여부
            if (sensorManager.getDefaultSensor(sensorType) == null) continue
            //측정하기 원하는 센서의 여부
            if (!list.contains(sensorType)) continue
            sensorManager.registerListener(
                listener,
                sensorManager.getDefaultSensor(sensorType),
                SensorManager.SENSOR_DELAY_NORMAL

            )
            sensorList.add(sensorType)
        }
        return sensorList
    }

    fun unregisterSensor(listener: SensorEventListener, list: ArrayList<Int>){
        sensorManager.unregisterListener(listener)
        list.clear()
    }
}