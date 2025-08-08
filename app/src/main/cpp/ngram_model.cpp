#include <regex>
#include <chrono>
#include <stdexcept>

#include "ngram_model.h"
#include "jni_log.h"

// NGramModel成员函数实现（仅修改参数访问方式）
std::vector<std::string> NGramModel::preprocess_text(const std::string &text) {
    std::vector<std::string> words;
    if (text.empty()) return words;

    // 转换为小写并替换标点符号
    std::string clean_text;
    clean_text.reserve(text.size());
    for (char c: text) {
        if (isalpha(c) || isdigit(c) || c == '\'' || c == ' ') {
            clean_text += tolower(c);
        } else {
            clean_text += ' ';
        }
    }

    // 分词
    std::stringstream ss(clean_text);
    std::string word;
    while (ss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }

    return words;
}

std::vector<std::vector<std::string>> NGramModel::build_ngrams(
        const std::vector<std::string> &words, int n) {

    std::vector<std::vector<std::string>> ngrams;
    if (words.size() < (size_t) n) return ngrams;

    ngrams.reserve(words.size() - n + 1);
    for (size_t i = 0; i <= words.size() - n; ++i) {
        std::vector<std::string> ngram(words.begin() + i, words.begin() + i + n);
        ngrams.push_back(ngram);
    }

    return ngrams;
}

void NGramModel::train(const std::string &text) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> words = preprocess_text(text);
    if (words.empty()) return;

    // 更新词汇表和词频统计
    for (const auto &word: words) {
        data_.vocabulary.insert(word);
        data_.word_count[word]++;
    }
    data_.total_words += words.size();

    // 训练不同大小的n元语法模型
    for (int i = 2; i <= data_.n; ++i) {
        auto ngrams = build_ngrams(words, i);
        if (ngrams.empty()) continue;

        // 确保模型容器存在
        if (data_.models.find(i) == data_.models.end()) {
            data_.models[i] = std::unordered_map<std::vector<std::string>,
                    std::unordered_map<std::string, int>, VectorHash>();
        }

        // 统计n元语法出现次数
        for (const auto &ngram: ngrams) {
            std::vector<std::string> context(ngram.begin(), ngram.end() - 1);
            const std::string &word = ngram.back();

            data_.models[i][context][word]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOGD("Training completed in %f seconds", elapsed.count());
}

std::vector<std::pair<std::string, double>> NGramModel::predict_next_word(
        const std::string &context, int num_predictions) {

    auto words = preprocess_text(context);
    std::unordered_map<std::string, double> candidates;

    // 如果没有上下文，返回最常见的词
    if (words.empty()) {
        std::vector<std::pair<std::string, int>> common_words;
        for (const auto &entry: data_.word_count) {
            common_words.emplace_back(entry.first, entry.second);
        }

        std::sort(common_words.begin(), common_words.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        std::vector<std::pair<std::string, double>> result;
        int total = data_.total_words > 0 ? data_.total_words : 1;
        for (size_t i = 0; i < common_words.size() && i < (size_t) num_predictions; ++i) {
            double prob = static_cast<double>(common_words[i].second) / total;
            result.emplace_back(common_words[i].first, prob);
        }
        return result;
    }

    // 尝试使用最大可能的n元模型
    int max_n = std::min(data_.n, (int) words.size() + 1);
    for (int n_size = max_n; n_size >= 2; --n_size) {
        int context_size = n_size - 1;
        std::vector<std::string> context_words(
                words.end() - context_size, words.end());

        auto it = data_.models.find(n_size);
        if (it == data_.models.end()) continue;

        auto &context_map = it->second;
        auto ctx_it = context_map.find(context_words);
        if (ctx_it == context_map.end()) continue;

        // 计算概率
        auto &word_counts = ctx_it->second;
        int total = std::accumulate(word_counts.begin(), word_counts.end(), 0,
                                    [](int sum, const auto &entry) { return sum + entry.second; });
        int vocab_size = data_.vocabulary.size();

        for (const auto &entry: word_counts) {
            double prob = (entry.second + data_.smoothing) /
                          (total + data_.smoothing * vocab_size);
            candidates[entry.first] += prob;
        }

        if (candidates.size() >= (size_t) num_predictions) {
            break;
        }
    }

    // 如果预测不够，使用一元模型补充
    if (candidates.size() < (size_t) num_predictions) {
        int remaining = num_predictions - candidates.size();
        int total = data_.total_words > 0 ? data_.total_words : 1;
        int vocab_size = data_.vocabulary.size() > 0 ? data_.vocabulary.size() : 1;

        std::vector<std::pair<std::string, int>> common_words;
        for (const auto &entry: data_.word_count) {
            if (candidates.find(entry.first) == candidates.end()) {
                common_words.emplace_back(entry.first, entry.second);
            }
        }

        std::sort(common_words.begin(), common_words.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        for (size_t i = 0; i < common_words.size() && i < (size_t) remaining; ++i) {
            double prob = (common_words[i].second + data_.smoothing) /
                          (total + data_.smoothing * vocab_size);
            candidates[common_words[i].first] = prob;
        }
    }

    // 排序并返回结果
    std::vector<std::pair<std::string, double>> result;
    for (const auto &entry: candidates) {
        result.push_back(entry);
    }

    std::sort(result.begin(), result.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    if (result.size() > (size_t) num_predictions) {
        result.resize(num_predictions);
    }

    return result;
}

// TextPredictor实现（保持不变）
TextPredictor::TextPredictor(const std::string &model_path, int n,
                             const std::vector<std::string> *sample_texts)
        : model_path_(model_path) {

    LOGD("Initializing predictor with model path: %s", model_path.c_str());

    // 检查模型文件是否存在
    std::ifstream ifs(model_path);
    if (ifs.good()) {
        LOGD("Loading existing model...");
        model_ = std::make_unique<NGramModel>();
        if (!model_->load(model_path)) {
            LOGE("Failed to load model, creating new one");
            model_ = std::make_unique<NGramModel>(n);
        }
    } else {
        LOGD("Creating new model with n=%d", n);
        model_ = std::make_unique<NGramModel>(n);

        // 如果提供了样本文本，进行预训练
        if (sample_texts && !sample_texts->empty()) {
            LOGD("Training with %zu sample texts", sample_texts->size());
            for (size_t i = 0; i < sample_texts->size(); ++i) {
                LOGD("Training sample %zu/%zu", i + 1, sample_texts->size());
                model_->train((*sample_texts)[i]);
            }
            save_model();
        }
    }
}

void TextPredictor::add_to_history(const std::string &text) {
    user_history_.push_back(text);
    LOGD("Added to history. Current size: %zu/%d",
         user_history_.size(), HISTORY_THRESHOLD);

    if (user_history_.size() >= HISTORY_THRESHOLD) {
        LOGD("History threshold reached, training model...");
        force_training();
    }
}

std::vector<std::pair<std::string, double>> TextPredictor::predict(
        const std::string &context, int num_predictions) {

    LOGD("Predicting for context: %s", context.c_str());
    return model_->predict_next_word(context, num_predictions);
}

bool TextPredictor::save_model() {
    if (model_) {
        return model_->save(model_path_);
    }
    return false;
}

bool TextPredictor::force_training() {
    if (user_history_.empty()) {
        LOGD("No history to train on");
        return false;
    }

    LOGD("Training on %zu history entries", user_history_.size());
    std::string all_text;
    for (const auto &text: user_history_) {
        all_text += text + " ";
    }

    model_->train(all_text);
    bool saved = save_model();
    user_history_.clear();
    return saved;
}

void TextPredictor::clear_history() {
    size_t count = user_history_.size();
    user_history_.clear();
    LOGD("Cleared %zu history entries", count);
}

std::string TextPredictor::get_model_info() const {
    if (!model_) return "No model available";

    std::stringstream ss;
    ss << "n: " << model_->get_model_data().n << "\n"
       << "Vocabulary size: " << model_->get_model_data().vocabulary.size() << "\n"
       << "Total words: " << model_->get_model_data().total_words << "\n"
       << "History entries: " << user_history_.size() << "\n"
       << "Smoothing: " << model_->get_model_data().smoothing;
    return ss.str();
}
