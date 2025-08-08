#ifndef NGRAM_MODEL_H
#define NGRAM_MODEL_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "ngram_model_io.h"

// N元语法模型类
class NGramModel {
private:
    NGramModelData data_;  // 封装的模型参数

    // 文本预处理和分词
    std::vector<std::string> preprocess_text(const std::string &text);

    // 生成n元语法
    std::vector<std::vector<std::string>>
    build_ngrams(const std::vector<std::string> &words, int n);

public:
    NGramModel(int n = 3, double smoothing = 0.1) {
        data_.n = n;
        data_.smoothing = smoothing;
    }

    // 训练模型
    void train(const std::string &text);

    // 预测下一个词
    std::vector<std::pair<std::string, double>> predict_next_word(
            const std::string &context, int num_predictions = 3);

    // 序列化相关方法（调用工具函数）
    bool save(const std::string &file_path) {
        return save_model_data(data_, file_path);
    }

    bool load(const std::string &file_path) {
        return load_model_data(data_, file_path);
    }

    auto get_model_data() {
        return data_;
    }
};

// 文本预测器类（保持不变）
class TextPredictor {
private:
    std::unique_ptr<NGramModel> model_;
    std::string model_path_;
    std::vector<std::string> user_history_;
    static const int HISTORY_THRESHOLD = 100;

public:
    TextPredictor(const std::string &model_path, int n = 3,
                  const std::vector<std::string> *sample_texts = nullptr);

    void add_to_history(const std::string &text);

    std::vector<std::pair<std::string, double>>
    predict(const std::string &context, int num_predictions = 3);

    bool save_model();

    bool force_training();

    void clear_history();

    std::string get_model_info() const;
};

#endif // NGRAM_MODEL_H
