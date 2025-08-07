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

// 哈希函数用于vector<string>作为unordered_map的键
struct VectorHash {
    size_t operator()(const std::vector<std::string> &v) const {
        std::hash<std::string> hasher;
        size_t seed = 0;
        for (const auto &s: v) {
            seed ^= hasher(s) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// N元语法模型类
class NGramModel {
private:
    int n_;
    double smoothing_;
    std::unordered_map<int, std::unordered_map<std::vector<std::string>,
            std::unordered_map<std::string, int>, VectorHash>> models_;
    std::unordered_map<std::string, int> word_count_;
    int total_words_;
    std::unordered_set<std::string> vocabulary_;

    // 文本预处理和分词
    std::vector<std::string> preprocess_text(const std::string &text);

    // 生成n元语法
    std::vector<std::vector<std::string>>
    build_ngrams(const std::vector<std::string> &words, int n);

public:
    NGramModel(int n = 3, double smoothing = 0.1);

    // 训练模型
    void train(const std::string &text);

    // 预测下一个词
    std::vector<std::pair<std::string, double>> predict_next_word(
            const std::string &context, int num_predictions = 3);

    // 序列化相关方法
    bool save(const std::string &file_path);

    bool load(const std::string &file_path);

    // Getters
    int get_n() const { return n_; }

    double get_smoothing() const { return smoothing_; }

    int get_vocabulary_size() const { return vocabulary_.size(); }

    int get_total_words() const { return total_words_; }
};

// 文本预测器类
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
