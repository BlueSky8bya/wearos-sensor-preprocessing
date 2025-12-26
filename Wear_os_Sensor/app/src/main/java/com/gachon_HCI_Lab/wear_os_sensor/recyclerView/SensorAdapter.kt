package com.gachon_HCI_Lab.wear_os_sensor.recyclerView

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CheckBox
import androidx.recyclerview.widget.RecyclerView
import com.example.wear_os_sensor_v2.R
import com.example.wear_os_sensor.SensorListUtil

class SensorAdapter(private var items: List<Item>) : RecyclerView.Adapter<SensorAdapter.ViewHolder>() {

    inner class ViewHolder(view: View) : RecyclerView.ViewHolder(view){
        val checkbox: CheckBox = itemView.findViewById(R.id.sensor_num)
    }

    private val sensorListUtil: SensorListUtil = SensorListUtil()

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.sensor_items, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        //item에 해당 차례의 checkbox 상태값 할당
        val item: Item = items[position]
        //현재 checkbox의 리스너를 null로 만듬
        holder.checkbox.setOnCheckedChangeListener(null)
        //현재 build되는 checkbox가 check되어 있으면 체크 표시
        holder.checkbox.isChecked = item.checked
        holder.checkbox.text = sensorListUtil.getSensorName(item.id)
        //상태값 업데이트
        holder.checkbox.setOnCheckedChangeListener{ _ , isChecked ->
            item.checked = isChecked
        }
    }

    override fun getItemCount(): Int{
        return items.size
    }
}