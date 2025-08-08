package com.tokyonth.textpredictor

import android.util.Pair

class TextPredictorNative @JvmOverloads constructor(
    modelPath: String,
    n: Int = 3,
    sampleTexts: Array<String>?,
) {

    companion object {
        init {
            System.loadLibrary("predictor")
        }
    }

    val predictorId: Long

    init {
        this.predictorId = createPredictor(modelPath, n, sampleTexts)
    }

    external fun createPredictor(modelPath: String, n: Int, sampleTexts: Array<String>?): Long

    external fun addToHistory(predictorId: Long, text: String)

    external fun predict(
        predictorId: Long,
        context: String,
        numPredictions: Int,
    ): Array<Pair<String, Double>>

    external fun forceTraining(predictorId: Long): Boolean

    external fun clearHistory(predictorId: Long)

    external fun getModelInfo(predictorId: Long): String

    external fun destroyPredictor(predictorId: Long)

    external fun isEnableLogging(isEnable: Boolean)

}
