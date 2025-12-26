package com.gachon_HCI_Lab.wear_os_sensor.util.connect

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothSocket
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import com.gachon_HCI_Lab.wear_os_sensor.util.step.StepsReaderUtil
import java.io.IOException
import java.io.OutputStream
import java.lang.reflect.Method
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*

@SuppressLint("StaticFieldLeak")
object BluetoothConnect {
    private var bluetoothManager: BluetoothManager? = null
    private lateinit var mBluetoothAdapter: BluetoothAdapter
    private lateinit var pairedDvices: Set<BluetoothDevice>
    private lateinit var mBluetoothDevice: BluetoothDevice
    private lateinit var mBluetoothSocket: BluetoothSocket
    private lateinit var mOutputStream: OutputStream
    private lateinit var context: Context
    var deviceName: String = ""

    fun constructor(context: Context){
        BluetoothConnect.context = context
        bluetoothManager = context.getSystemService(Context.BLUETOOTH_SERVICE) as BluetoothManager
        mBluetoothAdapter = bluetoothManager!!.adapter
    }
    @SuppressLint("MissingPermission")
    fun searchDevice(): String {
        pairedDvices = mBluetoothAdapter.bondedDevices
        val connected = getParingBluetoothDevice(pairedDvices) ?: return "error"
        mBluetoothDevice = mBluetoothAdapter.getRemoteDevice(connected.toString())
        val uuid = UUID.fromString("00001101-0000-1000-8000-00805f9b34fb")
        try {
            mBluetoothSocket = mBluetoothDevice.createRfcommSocketToServiceRecord(uuid)
        }catch (e: IOException) {
            Log.e("Search Device", "Error", e)
            return "error"
        }
        deviceName = connected.name
        return connected.name
    }

    @SuppressLint("MissingPermission")
    @Throws(IOException::class)
    fun connect(): String {
        try {
            Thread.sleep(100)
            mBluetoothSocket.connect()
            mOutputStream = mBluetoothSocket.outputStream
            return "Success"
        } catch (e: IOException) {
            throw IOException()
        }
    }

    @Throws(Exception::class)
    fun sendData(byteSensorData: ByteBuffer){
        // + battery 4바이트
        val byteBuffer = ByteBuffer.allocate(968)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        val byteArray = createByteArrayForSensorAndBatteryData(byteBuffer, byteSensorData)
        mOutputStream.write(byteArray)
        handleBluetoothError()
    }

    fun disconnect() {
        try {
            mBluetoothSocket.close()
            Log.i("Bluetooth Connection", "Error closing socket")
        } catch (e: IOException) {
            Log.e("Bluetooth Connection", "Error closing socket", e)
        }
    }

    //페어링된 기기를 찾는 함수
    @SuppressLint("MissingPermission")
    private fun getParingBluetoothDevice(pairedDevice: Set<BluetoothDevice>): BluetoothDevice? {
        try {
            for (bluetoothDevice: BluetoothDevice in pairedDevice) {
                if (isConnected(bluetoothDevice) as Boolean) {
                    Log.d(bluetoothDevice.toString(), bluetoothDevice.name)
                    return bluetoothDevice
                }
            }
        } catch (e: IOException) {
            //블루투스 서비스 사용불가인 경우
        }
        return null
    }
    private fun createByteArrayForSensorAndBatteryData(byteBuffer:ByteBuffer, byteSensorData: ByteBuffer): ByteArray{
        byteBuffer.putInt(getBatteryPercentage())
        byteBuffer.putInt(StepsReaderUtil.stepCount.toInt())
        byteBuffer.put(byteSensorData)
        return byteBuffer.array()
    }
    // 현재 연결되어있는지 유무
    private fun isConnected(device: BluetoothDevice): Any? {
        val m: Method = device.javaClass.getMethod("isConnected")
        return m.invoke(device)
    }

    private fun getBatteryPercentage(): Int {
        val batteryIntent =
            context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = batteryIntent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        return level * 100 / scale
    }

    private fun handleBluetoothError(){
        if(!mBluetoothSocket.isConnected){
            Log.i("Bluetooth Connection", mBluetoothSocket.isConnected.toString())
            throw Exception()
        }
    }
}