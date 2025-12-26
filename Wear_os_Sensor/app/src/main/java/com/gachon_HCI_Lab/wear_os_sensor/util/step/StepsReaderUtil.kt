package com.gachon_HCI_Lab.wear_os_sensor.util.step

import android.content.Context
import android.util.Log
import androidx.core.content.ContextCompat
import com.google.android.libraries.healthdata.data.ReadAggregatedDataResponse
import com.google.android.libraries.healthdata.permission.Permission
import com.google.common.util.concurrent.FutureCallback
import com.google.common.util.concurrent.Futures
import com.google.common.util.concurrent.ListenableFuture
import javax.annotation.Nullable
import javax.annotation.ParametersAreNonnullByDefault


object StepsReaderUtil {
    private val APP_TAG = "StepCount"
    private lateinit var context: Context
    var stepCount:Long = 0L

    fun addContext(context: Context){
        this.context = context
    }

    private fun permissionsExceptionHandler(exception: PermissionsException) {
        Log.e(APP_TAG, "PermissionsException with message: " + exception.message)
    }

    private fun onPermissionsFailureHandler(t: Throwable) {
        Log.e(APP_TAG, "Callback failed with message: $t")
    }

    fun readStepsWithPermissionsCheck() {
        try {
            val permissionFuture: ListenableFuture<Set<Permission>> =
                Permissions.getGrantedPermissions()
            Futures.addCallback<Set<Permission>>(
                permissionFuture, object : FutureCallback<Set<Permission?>?> {
                    override fun onSuccess(@Nullable result: Set<Permission?>?) {
                        if (Permissions.arePermissionsGranted(result as Set<Permission>?)) {
                            Log.d(APP_TAG, "All permissions granted. Read steps.")
                            readSteps()
                        } else {
                            Log.d(APP_TAG, "Permissions not granted. Request Permissions.")
                            readStepsWithRequestPermissions()
                        }
                    }

                    @ParametersAreNonnullByDefault
                    override fun onFailure(t: Throwable) {
                        onPermissionsFailureHandler(t)
                    }
                },
                ContextCompat.getMainExecutor(context /*context*/)
            )
        } catch (exception: PermissionsException) {
            permissionsExceptionHandler(exception)
        }
    }

    private fun readStepsWithRequestPermissions() {
        try {
            val requestPermissionFuture: ListenableFuture<Set<Permission>> =
                Permissions.requestPermissions()
            Futures.addCallback<Set<Permission>>(
                requestPermissionFuture, object : FutureCallback<Set<Permission?>?> {
                    override fun onSuccess(@Nullable result: Set<Permission?>?) {
                        if (Permissions.arePermissionsGranted(result as Set<Permission>?)) {
                            Log.d(APP_TAG, "All permissions granted. Read steps.")
                            readSteps()
                        } else {
                            Log.e(APP_TAG, "Permissions not granted. Can't read steps.")
                        }
                    }

                    @ParametersAreNonnullByDefault
                    override fun onFailure(t: Throwable) {
                        onPermissionsFailureHandler(t)
                    }
                },
                ContextCompat.getMainExecutor(context /*context*/)
            )
        } catch (exception: PermissionsException) {
            permissionsExceptionHandler(exception)
        }
    }

    fun readSteps() {
        try {
            val readFuture: ListenableFuture<ReadAggregatedDataResponse> =
                StepsReader.readAggregatedData()
            Futures.addCallback(
                readFuture, object : FutureCallback<ReadAggregatedDataResponse> {
                    override fun onSuccess(@Nullable result: ReadAggregatedDataResponse?) {
                        if(result == null)
                            return
                        stepCount = StepsReader.readStepsFromAggregatedDataResponse(result)
                    }
                    @ParametersAreNonnullByDefault
                    override fun onFailure(t: Throwable) {
                        Log.e(APP_TAG, "readAggregatedData Callback failed with message: $t")
                    }
                },
                ContextCompat.getMainExecutor(context /*context*/)
            )
        } catch (exception: StepsReaderException) {
            Log.e(APP_TAG, "StepsReaderException with message: " + exception.message)
        }
    }
}