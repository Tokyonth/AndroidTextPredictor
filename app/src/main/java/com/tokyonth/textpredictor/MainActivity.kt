package com.tokyonth.textpredictor

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.SeekBar
import androidx.core.widget.doOnTextChanged
import com.tokyonth.textpredictor.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private lateinit var textPredictionManager: TextPredictionManager

    private var predictionCount = 3

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        textPredictionManager = TextPredictionManager(this)

        binding.tvInfo.text = buildString {
            append("Model Info:")
            append("\n")
            append(textPredictionManager.getModelInfo())
        }
        binding.etInput.doOnTextChanged { text, _, _, _ ->
            if (text?.toString()?.endsWith(" ") == true) {
                val predictions =
                    textPredictionManager.predictNextWords(text.toString(), predictionCount)
                binding.tvPredictor.text = predictions.joinToString(", ")
            }
        }
        binding.seekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(
                seekBar: SeekBar?,
                progress: Int,
                fromUser: Boolean,
            ) {
                predictionCount = progress
                binding.tvNum.text = "Predictions count: $progress"
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
        binding.seekBar.progress = predictionCount
    }

    override fun onDestroy() {
        super.onDestroy()
        textPredictionManager.destroy()
    }

}
