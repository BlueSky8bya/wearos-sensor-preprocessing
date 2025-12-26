package com.gachon_HCI_Lab.wear_os_sensor

import android.app.ActivityManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Context.ACTIVITY_SERVICE
import android.content.Intent
import android.content.IntentFilter
import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.wear_os_sensor_v2.R
import com.example.wear_os_sensor_v2.databinding.FragmentConnectBinding
import com.gachon_HCI_Lab.wear_os_sensor.model.Constant
import com.gachon_HCI_Lab.wear_os_sensor.util.connect.BluetoothConnect


class ConnectFragment : Fragment() {

    private lateinit var binding: FragmentConnectBinding
    private lateinit var dataSender: BluetoothConnect

    // Fragment에서 LocalBroadcastManager를 사용하여 종료 메시지를 수신하고 UI를 업데이트.
    private val broadcastReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == "com.example.ACTION_SERVICE_STOPPED") {
                Log.i("serviceCheck", intent.hasExtra("state").toString())
                binding.startButton.text = "Start"
                // Service가 종료되었으므로 UI 업데이트 작업 수행
                if (intent.hasExtra("state")) {
                    binding.statusText.text = "Please start mobile app"
                    Toast.makeText(requireContext(), "Please start mobile app", Toast.LENGTH_SHORT)
                        .show()
                    return
                }
                binding.statusText.text = "Service is terminated"

            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = FragmentConnectBinding.inflate(inflater, container, false)
        val fragmentManager = requireActivity().supportFragmentManager
        val mainFragment = MainFragment()
        val view = binding.root

        dataSender = BluetoothConnect
        if (!isLocationServiceRunning()) {
            dataSender.constructor(requireContext())
            searchBluetoothDevice()
        } else {
            val name = dataSender.deviceName
            binding.connectText.text = "Device: $name"
            binding.statusText.text = "Service is Running"
            binding.startButton.text = "Stop"
        }

        binding.startButton.setOnClickListener {
            if (binding.startButton.text.toString() == "Start") {
                startLocationService()
            } else if (binding.startButton.text.toString() == "Stop") {
                fragmentManager.beginTransaction()
                    .replace(R.id.fragment_container, mainFragment)
                    .commit()
                stopLocationService()
                dataSender.disconnect()
            }
            else {
                Toast.makeText(requireActivity(), "Search device", Toast.LENGTH_SHORT).show()
                searchBluetoothDevice()
            }
        }

        // 중간에 끊길경우 예외 처리하기

        binding.stopButton.setOnClickListener {
            fragmentManager.beginTransaction()
                .replace(R.id.fragment_container, mainFragment)
                .commit()
            // 서비스 멈추기
            stopLocationService()
            if (binding.startButton.text.toString() == "Stop") {
                dataSender.disconnect()
            }
        }
        return view
    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter("com.example.ACTION_SERVICE_STOPPED")
        LocalBroadcastManager.getInstance(requireContext())
            .registerReceiver(broadcastReceiver, filter)
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(requireContext()).unregisterReceiver(broadcastReceiver)
    }

    private fun searchBluetoothDevice() {
        val device = dataSender.searchDevice()
        if (device != "error" && device != null) {
            Toast.makeText(requireActivity(), "Connect $device", Toast.LENGTH_SHORT).show()
            binding.connectText.text = "Device: $device"
            binding.statusText.text = "Ready to service"
            binding.startButton.text = "Start"
        } else
            binding.startButton.text = "Search"
    }

    private fun isLocationServiceRunning(): Boolean {
        val activityManager =
            requireActivity().getSystemService(ACTIVITY_SERVICE) as ActivityManager
        if (activityManager != null) {
            for (service in activityManager.getRunningServices(Int.MAX_VALUE)) {
                if (SensorService::class.java.name == service.service.className) {
                    if (service.foreground) {
                        return true
                    }
                }
            }
        }
        return false
    }

    private fun startLocationService() {
        if (!isLocationServiceRunning()) {
            val intent = Intent(requireContext(), SensorService::class.java)
            intent.action = Constant.ACTION_START_LOCATION_SERVICE
            requireActivity().startService(intent)
            updateTextStatus()
            Toast.makeText(requireContext(), "Service started", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopLocationService() {
        if (isLocationServiceRunning()) {
            val intent = Intent(requireContext(), SensorService::class.java)
            intent.action = Constant.ACTION_STOP_LOCATION_SERVICE
            requireActivity().stopService(intent)
            updateTextStatus()
            Toast.makeText(requireContext(), "Service stopped", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateTextStatus() {
        if (!isLocationServiceRunning()) {
            binding.statusText.text = "Service is Running"
            binding.startButton.text = "Stop"
        } else {
            binding.statusText.text = "Service is terminated"
            binding.startButton.text = "Start"
        }
    }
}