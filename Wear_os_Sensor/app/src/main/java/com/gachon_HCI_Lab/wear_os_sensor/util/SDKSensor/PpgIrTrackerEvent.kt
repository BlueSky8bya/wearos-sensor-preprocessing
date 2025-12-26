package com.gachon_HCI_Lab.wear_os_sensor.util.SDKSensor

import android.util.Log
import com.gachon_HCI_Lab.wear_os_sensor.model.SDKSensor
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.ValueKey

object PpgIrTrackerEvent: HealthTracker.TrackerEventListener {
    val TAG: String = "ppgIr"
    var count: Int = 0
    override fun onDataReceived(list: List<DataPoint?>) {
        if (list.size != 0) {
            for (dataPoint in list) {
                Thread{
                    SDKSensor.ppgIrList.add(count.toString() + " - Ppg Ir : " + dataPoint!!.getValue(ValueKey.PpgIrSet.PPG_IR) + "\n")
                }.start()

            }

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

}