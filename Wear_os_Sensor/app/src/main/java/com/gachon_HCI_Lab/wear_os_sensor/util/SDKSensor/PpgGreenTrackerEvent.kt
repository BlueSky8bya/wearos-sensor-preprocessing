package com.gachon_HCI_Lab.wear_os_sensor.util.SDKSensor

import android.util.Log
import com.gachon_HCI_Lab.wear_os_sensor.model.SensorModel
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.ValueKey
import java.nio.ByteBuffer
import java.nio.ByteOrder

object PpgGreenTrackerEvent : HealthTracker.TrackerEventListener {
    val TAG: String = "ppgGreen"
    override fun onDataReceived(list: List<DataPoint?>) {
        if (list.size != 0) {
            Thread {
                for (dataPoint in list) {
                    val byteData = createByteBuffer(dataPoint)
                    SensorModel.sendData.offer(byteData)
                }
            }.start()
        } else {
            Log.i(TAG, "onDataReceived List is zero")
        }
    }

    override fun onFlushCompleted() {
        Log.i(TAG, " onFlushCompleted called")
    }

    override fun onError(trackerError: HealthTracker.TrackerError) {
        Log.i(TAG, " onError called")
        if (trackerError === HealthTracker.TrackerError.PERMISSION_ERROR) {
            //
        }
        if (trackerError === HealthTracker.TrackerError.SDK_POLICY_ERROR) {
            //
        }
    }

    private fun createByteBuffer(dataPoint: DataPoint?): ByteBuffer {
        val byteBuffer = ByteBuffer.allocate(16)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        byteBuffer.putInt(30)
        byteBuffer.putLong(dataPoint!!.getTimestamp())
        byteBuffer.putFloat(dataPoint!!.getValue(ValueKey.PpgGreenSet.PPG_GREEN).toFloat())
        byteBuffer.position(0)
        Log.i("ppgGreen", dataPoint!!.getTimestamp().toString())
        Log.i("ppgGreen", dataPoint!!.getValue(ValueKey.PpgGreenSet.PPG_GREEN).toFloat().toString())
        return byteBuffer
    }
}