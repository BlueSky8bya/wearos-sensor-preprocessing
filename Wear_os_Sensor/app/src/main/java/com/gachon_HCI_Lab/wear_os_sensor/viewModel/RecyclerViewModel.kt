package com.example.wear_os_sensor

import android.util.Log
import com.gachon_HCI_Lab.wear_os_sensor.model.SensorModel
import com.gachon_HCI_Lab.wear_os_sensor.recyclerView.Item
import com.gachon_HCI_Lab.wear_os_sensor.recyclerView.SensorAdapter

object RecyclerViewModel {
    private val sensorListUtil: SensorListUtil = SensorListUtil()

    fun createSensorList(){
        var items = SensorModel.getCheckedSensorList()
        if (items.size > 0) return
        for (name in sensorListUtil.getSensorIDList()){
            Log.d("sesnsor lsit", name.toString())
            if(name == 1 || name == 18 || name == 4 || name == 5 || name == 21 || name == 9) continue
            items.add(Item(name, true))
        }
    }

    fun getSensorAdapter(): SensorAdapter {
        var items = SensorModel.getCheckedSensorList()
        return SensorAdapter(items)
    }
}