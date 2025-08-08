package com.tokyonth.textpredictor

import android.content.Context
import java.io.File

class TextPredictionManager(
    private val context: Context,
) {

    // 获取模型存储路径（应用私有目录）
    private val modelPath = "${context.filesDir}/ngram_model.bin"

    // 创建预测器实例
    private val predictor: TextPredictorNative

    init {
        val dataSet = if (File(modelPath).exists()) {
            null
        } else {
            getSampleTexts()
        }
        predictor = TextPredictorNative(modelPath, 3, dataSet)
    }

    // 初始化样例文本（用于首次训练）
    private fun getSampleTexts(): Array<String> {
        context.assets.open("pod_dataset.txt").use {
            return it.bufferedReader().readLines().toTypedArray()
        }
    }

    /**
     * 添加用户输入到历史记录，达到阈值时自动训练
     */
    fun addUserInput(text: String) {
        if (text.isNotBlank()) {
            predictor.addToHistory(predictor.predictorId, text)
        }
    }

    /**
     * 预测下一个可能的词
     * @param context 当前输入的上下文文本
     * @param count 希望返回的预测数量
     * @return 预测的词及其概率（降序排列）
     */
    fun predictNextWords(context: String, count: Int = 3): List<String> {
        return try {
            val predictions = predictor.predict(predictor.predictorId, context, count)
            predictions.mapNotNull { it.first }
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList()
        }
    }

    /**
     * 强制立即训练模型（不等待历史记录达到阈值）
     */
    fun forceTrain() {
        predictor.forceTraining(predictor.predictorId)
    }

    /**
     * 清除用户历史记录
     */
    fun clearHistory() {
        predictor.clearHistory(predictor.predictorId)
    }

    /**
     * 获取模型信息（调试用）
     */
    fun getModelInfo(): String {
        return predictor.getModelInfo(predictor.predictorId)
    }

    /**
     * 释放资源
     */
    fun destroy() {
        predictor.destroyPredictor(predictor.predictorId)
    }

    fun setIsEnableLogging(isEnable: Boolean) {
        predictor.isEnableLogging(isEnable)
    }

}
