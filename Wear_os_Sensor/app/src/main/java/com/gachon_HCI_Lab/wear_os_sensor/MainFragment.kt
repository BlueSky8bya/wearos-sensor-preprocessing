package com.gachon_HCI_Lab.wear_os_sensor

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentTransaction
import com.example.wear_os_sensor_v2.R
import com.example.wear_os_sensor.RecyclerViewModel

class MainFragment : Fragment() {

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_main, container, false)
        RecyclerViewModel.createSensorList()
        val fragmentManager: FragmentManager = requireActivity().supportFragmentManager
        val fragmentTransaction: FragmentTransaction = fragmentManager.beginTransaction()

        // 버튼 클릭 이벤트 처리
        val button1 = view.findViewById<View>(R.id.sensors)
        button1.setOnClickListener {
            // 다른 프래그먼트로 전환
            val sensorListFragment = SensorListFragment()
            fragmentTransaction.replace(R.id.fragment_container, sensorListFragment).commit()
        }

        val button2 = view.findViewById<View>(R.id.start_button)
        button2.setOnClickListener {
            val startFragment = ConnectFragment()
            fragmentTransaction.replace(R.id.fragment_container, startFragment).commit()
        }

        return view
    }
}