package com.gachon_HCI_Lab.wear_os_sensor.util.step

import com.google.android.libraries.healthdata.HealthDataClient
import com.google.android.libraries.healthdata.data.*
import com.google.common.util.concurrent.ListenableFuture
import org.jetbrains.annotations.Nullable
import java.time.LocalDateTime
import java.time.LocalTime


object StepsReader {
    private lateinit var healthDataClient: HealthDataClient

    fun addHealthDataClient(healthDataClient: HealthDataClient){
        this.healthDataClient = healthDataClient
    }

    @Throws(StepsReaderException::class)
    fun readAggregatedData(): ListenableFuture<ReadAggregatedDataResponse> {
        if (healthDataClient == null) {
            throw StepsReaderException("health client is null")
        }
        /*******************************************************************************************
         * [Practice 2] Build read aggregated data request object for read today's steps
         * - Set interval data type of steps
         * -------------------------------------------------------------------------------------------
         * - (Hint) Uncomment line 42 and fill below TODO 2 with
         * (1) for interval data type: IntervalDataTypes.STEPS
         */
        val readAggregatedDataRequest = ReadAggregatedDataRequest.builder()
            .setTimeSpec(
                TimeSpec.builder()
                    .setStartLocalDateTime(LocalDateTime.now().with(LocalTime.MIDNIGHT))
                    .build()
            ) .addCumulativeAggregationSpec(CumulativeAggregationSpec.builder(IntervalDataTypes.STEPS).build())
            .build()
        return healthDataClient.readAggregatedData(readAggregatedDataRequest)
    }

    fun readStepsFromAggregatedDataResponse(@Nullable result: ReadAggregatedDataResponse?): Long {
        /*******************************************************************************************
         * [Practice 3] Read aggregated value from cumulative data and add them to the result
         * - Get AggregatedValue from cumulativeData object
         * - Get steps count from AggregatedValue object
         * -------------------------------------------------------------------------------------------
         * - (Hint) Replace TODO 3 with parts of code
         * (1) get AggregatedValue object 'obj' using cumulativeData.getTotal()
         * (2) get value using obj.getLongValue() and add to the result
         */
        var steps = 0L
        if (result != null) {
            val cumulativeDataList = result.cumulativeDataList
            if (cumulativeDataList.isNotEmpty()) {
                for (cumulativeData in cumulativeDataList) {
                    if(cumulativeData.total == null) {
                        steps = 0L
                        continue
                    }
                    steps = cumulativeData.total?.longValue ?: 0L
                }
            }
        }
        return steps
    }
}