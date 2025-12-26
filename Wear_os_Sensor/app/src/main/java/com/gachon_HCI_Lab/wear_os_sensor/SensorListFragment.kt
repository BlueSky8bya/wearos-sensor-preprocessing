package com.gachon_HCI_Lab.wear_os_sensor

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.FragmentTransaction
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.wear_os_sensor_v2.R
import com.example.wear_os_sensor.RecyclerViewModel
import com.example.wear_os_sensor_v2.databinding.FragmentSensorListBinding

class SensorListFragment : Fragment() {
    private lateinit var recyclerView: RecyclerView

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val binding = FragmentSensorListBinding.inflate(inflater, container, false)
        // SensorModel을 사용하여 RecyclerViewModel을 생성합니다.
        recyclerView = binding.sensorRecyclerView
        recyclerView.layoutManager = LinearLayoutManager(requireContext())
        recyclerView.adapter = RecyclerViewModel.getSensorAdapter()

        val fragmentManager: FragmentManager = requireActivity().supportFragmentManager
        val fragmentTransaction: FragmentTransaction = fragmentManager.beginTransaction()

        binding.back.setOnClickListener {
            val mainFragment = MainFragment()
            fragmentTransaction.replace(R.id.fragment_container, mainFragment).commit()
        }

        return binding.root
    }
}