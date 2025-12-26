//package com.gachon_HCI_Lab.wear_os_sensor
//
//import android.app.NotificationChannel
//import android.app.NotificationManager
//import android.app.PendingIntent
//import android.app.Service
//import android.content.Intent
//import android.hardware.Sensor
//import android.hardware.SensorEvent
//import android.hardware.SensorEventListener
//import android.hardware.SensorManager
//import android.os.Build
//import android.os.IBinder
//import android.util.Log
//import androidx.core.app.NotificationCompat
//import androidx.localbroadcastmanager.content.LocalBroadcastManager
//import com.example.wear_os_sensor.SensorViewModel
//import com.gachon_HCI_Lab.wear_os_sensor.model.Constant
//import com.gachon_HCI_Lab.wear_os_sensor.model.SensorModel
//import com.gachon_HCI_Lab.wear_os_sensor.util.PpgUtil
//import com.gachon_HCI_Lab.wear_os_sensor.util.connect.BluetoothConnect
//import com.gachon_HCI_Lab.wear_os_sensor.util.step.StepsReaderUtil
//import java.io.IOException
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//
//
//class SensorService : Service(), SensorEventListener {
//
//    private lateinit var sensorViewModel: SensorViewModel
//    private var dataSender: BluetoothConnect = BluetoothConnect
//    val intent = Intent("com.example.ACTION_SERVICE_STOPPED")
//    val ppg = PpgUtil(this)
//
//    override fun onBind(intent: Intent?): IBinder? {
//        TODO("Not yet implemented")
//    }
//
//    fun startForground() {
//        setForground()
//        sensorViewModel = SensorViewModel(getSystemService(SENSOR_SERVICE) as SensorManager, this)
//        //sensor 연결
//        sensorViewModel.register()
//        // 통신 연결
//        try {
//            val msg1: String = dataSender.connect()
//            Log.v("connect", msg1)
//        } catch (e: IOException) {
//            intent.putExtra("state", "mobile Error")
//            stopForground()
//            return
//        }
//
//        ppg.start()
//        Thread {
//            while (true) {
//                if (SensorModel.sendData.size >= 40) {
//                    var sendBinary = createSendData()
//                    StepsReaderUtil.readSteps()
//                    try {
//                        dataSender.sendData(sendBinary)
//                        Thread.sleep(500)
//                    } catch (e: Exception) {
//                        handleMobileError()
//                    }
//                }
//            }
//        }.start()
//    }
//
//    fun stopForground() {
//        ppg.destroy()
//        sensorViewModel.unRegister()
//        LocalBroadcastManager.getInstance(applicationContext).sendBroadcast(intent)
//        dataSender.disconnect()
//        stopSelf()
//    }
//
//    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
//        // 서비스가 시작될 때 호출되는 메소드입니다.
//        // 서비스 실행 코드 작성
//        if (intent != null) {
//            val action = intent.action
//            if (action != null) {
//                if (action == Constant.ACTION_START_LOCATION_SERVICE) {
//                    startForground()
//                } else if (action == Constant.ACTION_STOP_LOCATION_SERVICE) {
//                    stopForground()
//                }
//            }
//        }
//        return START_STICKY
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        ppg.destroy()
//        sensorViewModel.unRegister()
//        // Service에서 LocalBroadcastManager를 사용하여 종료 메시지를 보냅니다.
//        LocalBroadcastManager.getInstance(applicationContext).sendBroadcast(intent)
//        stopSelf()
//    }
//
//    override fun onSensorChanged(event: SensorEvent?) {
//        if (event == null) return
//        val byteData = sensorViewModel.sensorValue(event)
//        SensorModel.sendData.offer(byteData)
//    }
//
//    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
//        //
//    }
//
//    fun setForground() {
//        val builder: NotificationCompat.Builder
//
//        val notificationIntent = Intent(this, MainActivity::class.java)
//        notificationIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
//        val pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE)
//
//        builder = if (Build.VERSION.SDK_INT >= 26) {
//            val CHANNEL_ID = "measuring_service_channel"
//            val channel = NotificationChannel(
//                CHANNEL_ID,
//                "Measuring Service Channel",
//                NotificationManager.IMPORTANCE_DEFAULT
//            )
//            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
//                .createNotificationChannel(channel)
//            NotificationCompat.Builder(this, CHANNEL_ID)
//        } else {
//            NotificationCompat.Builder(this)
//        }
//
//        builder.setContentTitle("측정시작됨")
//            .setContentIntent(pendingIntent)
//
//        startForeground(1, builder.build())
//    }
//
//    private fun createSendData(): ByteBuffer {
//        val byteBuffer = ByteBuffer.allocate(960)
//        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
//        for (i in 0..39) {
//            val buffer = SensorModel.sendData.take()
//            byteBuffer.put(buffer)
//        }
//        byteBuffer.position(0)
//        return byteBuffer
//    }
//
//    private fun handleMobileError() {
//        SensorModel.sendData.clear()
//        ppg.destroy()
//        intent.putExtra("state", "mobile Error")
//        stopForground()
//    }
//}

package com.gachon_HCI_Lab.wear_os_sensor

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.IBinder
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.example.wear_os_sensor.SensorViewModel
import com.example.wear_os_sensor_v2.R
import com.gachon_HCI_Lab.wear_os_sensor.model.Constant
import com.gachon_HCI_Lab.wear_os_sensor.model.SensorModel
import com.gachon_HCI_Lab.wear_os_sensor.util.PpgUtil
import com.gachon_HCI_Lab.wear_os_sensor.util.connect.BluetoothConnect
import com.gachon_HCI_Lab.wear_os_sensor.util.step.StepsReaderUtil
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SensorService : Service(), SensorEventListener {

    private lateinit var sensorViewModel: SensorViewModel
    private var dataSender: BluetoothConnect = BluetoothConnect
    val intent = Intent("com.example.ACTION_SERVICE_STOPPED")
    val ppg = PpgUtil(this)

    // 🔹 진동기 객체
    private lateinit var vibrator: Vibrator

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onCreate() {
        super.onCreate()
        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        }
    }

    fun startForground() {
        setForground()
        sensorViewModel = SensorViewModel(getSystemService(SENSOR_SERVICE) as SensorManager, this)
        sensorViewModel.register()

        try {
            val msg1: String = dataSender.connect()
            Log.v("connect", msg1)
        } catch (e: IOException) {
            intent.putExtra("state", "mobile Error")
            handleMobileError()
            return
        }

        ppg.start()
        Thread {
            while (true) {
                if (SensorModel.sendData.size >= 40) {
                    val sendBinary = createSendData()
                    StepsReaderUtil.readSteps()
                    try {
                        dataSender.sendData(sendBinary)
                        Thread.sleep(500)
                    } catch (e: Exception) {
                        handleMobileError()
                    }
                }
            }
        }.start()
    }

    fun stopForground() {
        ppg.destroy()
        sensorViewModel.unRegister()
        LocalBroadcastManager.getInstance(applicationContext).sendBroadcast(intent)
        dataSender.disconnect()
        stopSelf()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent != null) {
            when (intent.action) {
                Constant.ACTION_START_LOCATION_SERVICE -> startForground()
                Constant.ACTION_STOP_LOCATION_SERVICE -> stopForground()
            }
        }
        return START_STICKY
    }

//    override fun onDestroy() {
//        super.onDestroy()
//        ppg.destroy()
//        sensorViewModel.unRegister()
//        LocalBroadcastManager.getInstance(applicationContext).sendBroadcast(intent)
//        stopSelf()
//    }

    override fun onDestroy() {
        super.onDestroy()
        vibrator.vibrate(VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE))

        val builder = NotificationCompat.Builder(this, "measuring_service_channel")
            .setContentTitle("서비스가 중단되었습니다")
            .setContentText("측정이 중지되었습니다. 다시 실행해주세요.")
            .setSmallIcon(R.drawable.component_19)
            .setPriority(NotificationCompat.PRIORITY_HIGH)

        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(2, builder.build())
    }


    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null) return
        val byteData = sensorViewModel.sensorValue(event)
        SensorModel.sendData.offer(byteData)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    fun setForground() {
        val builder: NotificationCompat.Builder
        val notificationIntent = Intent(this, MainActivity::class.java)
        notificationIntent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
        val pendingIntent = PendingIntent.getActivity(this, 0, notificationIntent, PendingIntent.FLAG_IMMUTABLE)

        builder = if (Build.VERSION.SDK_INT >= 26) {
            val CHANNEL_ID = "measuring_service_channel"
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Measuring Service Channel",
                NotificationManager.IMPORTANCE_DEFAULT
            )
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
            NotificationCompat.Builder(this, CHANNEL_ID)
        } else {
            NotificationCompat.Builder(this)
        }

        builder.setContentTitle("측정시작됨")
            .setContentIntent(pendingIntent)

        startForeground(1, builder.build())
    }

    private fun createSendData(): ByteBuffer {
        val byteBuffer = ByteBuffer.allocate(960)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0..39) {
            val buffer = SensorModel.sendData.take()
            byteBuffer.put(buffer)
        }
        byteBuffer.position(0)
        return byteBuffer
    }

    // 🔹 연결 끊김 처리 + 무한 재연결 + 재연결 성공 시 진동 + 재시작
    private fun handleMobileError() {
        SensorModel.sendData.clear()
        ppg.destroy()
        intent.putExtra("state", "mobile Error")
        LocalBroadcastManager.getInstance(applicationContext).sendBroadcast(intent)
        dataSender.disconnect()

        Thread {
            var success = false
            while (!success) {
                try {
                    val msg = dataSender.connect()
                    Log.v("reconnect", msg)
                    success = true
                } catch (e: IOException) {
                    Log.e("reconnect", "재연결 실패, 3초 후 재시도")
                    Thread.sleep(3000)
                }
            }

            // 재연결 성공 → 진동 + 측정 재시작
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                vibrator.vibrate(1000)
            }

            startForground()
        }.start()
    }
}
