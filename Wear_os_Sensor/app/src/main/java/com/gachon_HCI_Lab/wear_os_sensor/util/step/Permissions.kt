package com.gachon_HCI_Lab.wear_os_sensor.util.step

import com.google.android.libraries.healthdata.HealthDataClient
import com.google.android.libraries.healthdata.data.IntervalDataTypes
import com.google.android.libraries.healthdata.permission.AccessType
import com.google.android.libraries.healthdata.permission.Permission
import com.google.common.util.concurrent.ListenableFuture
import org.jetbrains.annotations.Nullable


object Permissions {
    private var healthDataClient: HealthDataClient? = null
    private val permissions: MutableSet<Permission> = HashSet()

    fun addHealthDataClietn(healthDataClient: HealthDataClient?) {
        this.healthDataClient = healthDataClient
        /*******************************************************************************************
         * [Practice 1] Build permission object grand permissions for read today's steps
         * - Set interval data type of steps
         * - Set read access type
         * -------------------------------------------------------------------------------------------
         * - (Hint) Uncomment lines 30,31 and fill below TODOs with
         * (1) for interval data type: IntervalDataTypes.STEPS
         * (2) for read access: AccessType.READ
         */
        val stepsReadPermission: Permission = Permission.builder()
            .setDataType(IntervalDataTypes.STEPS)
            .setAccessType(AccessType.READ)
            .build()
        permissions.add(stepsReadPermission)
    }

    @Throws(PermissionsException::class)
    fun getGrantedPermissions(): ListenableFuture<Set<Permission>> {
        if (healthDataClient == null) {
            throw PermissionsException("health client is null")
        }
        return healthDataClient!!.getGrantedPermissions(permissions)
    }


    @Throws(PermissionsException::class)
    fun requestPermissions(): ListenableFuture<Set<Permission>> {
        if (healthDataClient == null) {
            throw PermissionsException("health client is null")
        }
        return healthDataClient!!.requestPermissions(permissions)
    }

    fun arePermissionsGranted(@Nullable result: Set<Permission>?): Boolean {
        return result?.containsAll(permissions) ?: false
    }
}