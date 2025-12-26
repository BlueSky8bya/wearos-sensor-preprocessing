package com.gachon_HCI_Lab.wear_os_sensor

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.fragment.app.Fragment
import com.example.wear_os_sensor_v2.R
import com.example.wear_os_sensor_v2.databinding.FragmentConnectBinding
import java.util.*

// TODO: Rename parameter arguments, choose names that match
// the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
private const val ARG_PARAM1 = "param1"
private const val ARG_PARAM2 = "param2"

/**
 * A simple [Fragment] subclass.
 * Use the [ExceptFragment.newInstance] factory method to
 * create an instance of this fragment.
 */

class ExceptFragment : Fragment(){
    private lateinit var binding: FragmentConnectBinding

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_except, container, false)
        val button = view.findViewById<Button>(R.id.exceptButton)
        button.setOnClickListener {
            System.exit(10)
        }
        return view
    }
}