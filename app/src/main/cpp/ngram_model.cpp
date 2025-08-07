#include "ngram_model.h"
#include <regex>
#include <chrono>
#include <android/log.h>

#define LOG_TAG "NGramModel"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// 构造函数
NGramModel::NGramModel(int n, double smoothing)
        : n_(n), smoothing_(smoothing), total_words_(0) {}

// 文本预处理
std::vector<std::string> NGramModel::preprocess_text(const std::string& text) {
    std::vector<std::string> words;
    if (text.empty()) return words;

    // 转换为小写并替换标点符号
    std::string clean_text;
    clean_text.reserve(text.size());
    for (char c : text) {
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

// 生成n元语法
std::vector<std::vector<std::string>> NGramModel::build_ngrams(
        const std::vector<std::string>& words, int n) {

    std::vector<std::vector<std::string>> ngrams;
    if (words.size() < (size_t)n) return ngrams;

    ngrams.reserve(words.size() - n + 1);
    for (size_t i = 0; i <= words.size() - n; ++i) {
        std::vector<std::string> ngram(words.begin() + i, words.begin() + i + n);
        ngrams.push_back(ngram);
    }

    return ngrams;
}

// 训练模型
void NGramModel::train(const std::string& text) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::string> words = preprocess_text(text);
    if (words.empty()) return;

    // 更新词汇表和词频统计
    for (const auto& word : words) {
        vocabulary_.insert(word);
        word_count_[word]++;
    }
    total_words_ += words.size();

    // 训练不同大小的n元语法模型
    for (int i = 2; i <= n_; ++i) {
        auto ngrams = build_ngrams(words, i);
        if (ngrams.empty()) continue;

        // 确保模型容器存在
        if (models_.find(i) == models_.end()) {
            models_[i] = std::unordered_map<std::vector<std::string>,
                    std::unordered_map<std::string, int>, VectorHash>();
        }

        // 统计n元语法出现次数
        for (const auto& ngram : ngrams) {
            std::vector<std::string> context(ngram.begin(), ngram.end() - 1);
            const std::string& word = ngram.back();

            models_[i][context][word]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    LOGD("Training completed in %f seconds", elapsed.count());
}

// 预测下一个词
std::vector<std::pair<std::string, double>> NGramModel::predict_next_word(
        const std::string& context, int num_predictions) {

    auto words = preprocess_text(context);
    std::unordered_map<std::string, double> candidates;

    // 如果没有上下文，返回最常见的词
    if (words.empty()) {
        std::vector<std::pair<std::string, int>> common_words;
        for (const auto& entry : word_count_) {
            common_words.emplace_back(entry.first, entry.second);
        }

        std::sort(common_words.begin(), common_words.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        std::vector<std::pair<std::string, double>> result;
        int total = total_words_ > 0 ? total_words_ : 1;
        for (size_t i = 0; i < common_words.size() && i < (size_t)num_predictions; ++i) {
            double prob = static_cast<double>(common_words[i].second) / total;
            result.emplace_back(common_words[i].first, prob);
        }
        return result;
    }

    // 尝试使用最大可能的n元模型
    int max_n = std::min(n_, (int)words.size() + 1);
    for (int n_size = max_n; n_size >= 2; --n_size) {
        int context_size = n_size - 1;
        std::vector<std::string> context_words(
                words.end() - context_size, words.end());

        auto it = models_.find(n_size);
        if (it == models_.end()) continue;

        auto& context_map = it->second;
        auto ctx_it = context_map.find(context_words);
        if (ctx_it == context_map.end()) continue;

        // 计算概率
        auto& word_counts = ctx_it->second;
        int total = std::accumulate(word_counts.begin(), word_counts.end(), 0,
                                    [](int sum, const auto& entry) { return sum + entry.second; });
        int vocab_size = vocabulary_.size();

        for (const auto& entry : word_counts) {
            double prob = (entry.second + smoothing_) /
                          (total + smoothing_ * vocab_size);
            candidates[entry.first] += prob;
        }

        if (candidates.size() >= (size_t)num_predictions) {
            break;
        }
    }

    // 如果预测不够，使用一元模型补充
    if (candidates.size() < (size_t)num_predictions) {
        int remaining = num_predictions - candidates.size();
        int total = total_words_ > 0 ? total_words_ : 1;
        int vocab_size = vocabulary_.size() > 0 ? vocabulary_.size() : 1;

        std::vector<std::pair<std::string, int>> common_words;
        for (const auto& entry : word_count_) {
            if (candidates.find(entry.first) == candidates.end()) {
                common_words.emplace_back(entry.first, entry.second);
            }
        }

        std::sort(common_words.begin(), common_words.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < common_words.size() && i < (size_t)remaining; ++i) {
            double prob = (common_words[i].second + smoothing_) /
                          (total + smoothing_ * vocab_size);
            candidates[common_words[i].first] = prob;
        }
    }

    // 排序并返回结果
    std::vector<std::pair<std::string, double>> result;
    for (const auto& entry : candidates) {
        result.push_back(entry);
    }

    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    if (result.size() > (size_t)num_predictions) {
        result.resize(num_predictions);
    }

    return result;
}

// 保存模型到文件
bool NGramModel::save(const std::string& file_path) {
    try {
        std::ofstream ofs(file_path, std::ios::binary);
        if (!ofs.is_open()) return false;

        // 写入基本参数
        ofs.write(reinterpret_cast<const char*>(&n_), sizeof(n_));
        ofs.write(reinterpret_cast<const char*>(&smoothing_), sizeof(smoothing_));
        ofs.write(reinterpret_cast<const char*>(&total_words_), sizeof(total_words_));

        // 写入词汇表
        size_t vocab_size = vocabulary_.size();
        ofs.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        for (const auto& word : vocabulary_) {
            size_t len = word.size();
            ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
            ofs.write(word.c_str(), len);
        }

        // 写入词频统计
        size_t wc_size = word_count_.size();
        ofs.write(reinterpret_cast<const char*>(&wc_size), sizeof(wc_size));
        for (const auto& entry : word_count_) {
            size_t len = entry.first.size();
            ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
            ofs.write(entry.first.c_str(), len);
            ofs.write(reinterpret_cast<const char*>(&entry.second), sizeof(entry.second));
        }

        // 写入模型数据
        size_t model_size = models_.size();
        ofs.write(reinterpret_cast<const char*>(&model_size), sizeof(model_size));
        for (const auto& model_entry : models_) {
            int n_size = model_entry.first;
            ofs.write(reinterpret_cast<const char*>(&n_size), sizeof(n_size));

            const auto& context_map = model_entry.second;
            size_t context_size = context_map.size();
            ofs.write(reinterpret_cast<const char*>(&context_size), sizeof(context_size));

            for (const auto& context_entry : context_map) {
                const auto& context = context_entry.first;
                size_t ctx_len = context.size();
                ofs.write(reinterpret_cast<const char*>(&ctx_len), sizeof(ctx_len));

                for (const auto& word : context) {
                    size_t len = word.size();
                    ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
                    ofs.write(word.c_str(), len);
                }

                const auto& word_map = context_entry.second;
                size_t word_map_size = word_map.size();
                ofs.write(reinterpret_cast<const char*>(&word_map_size), sizeof(word_map_size));

                for (const auto& word_entry : word_map) {
                    size_t len = word_entry.first.size();
                    ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
                    ofs.write(word_entry.first.c_str(), len);
                    ofs.write(reinterpret_cast<const char*>(&word_entry.second), sizeof(word_entry.second));
                }
            }
        }

        return true;
    } catch (...) {
        LOGE("Error saving model");
        return false;
    }
}

// 从文件加载模型
bool NGramModel::load(const std::string& file_path) {
    try {
        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open()) return false;

        // 清空现有数据
        models_.clear();
        word_count_.clear();
        vocabulary_.clear();
        total_words_ = 0;

        // 读取基本参数
        ifs.read(reinterpret_cast<char*>(&n_), sizeof(n_));
        ifs.read(reinterpret_cast<char*>(&smoothing_), sizeof(smoothing_));
        ifs.read(reinterpret_cast<char*>(&total_words_), sizeof(total_words_));

        // 读取词汇表
        size_t vocab_size;
        ifs.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        for (size_t i = 0; i < vocab_size; ++i) {
            size_t len;
            ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string word(len, ' ');
            ifs.read(&word[0], len);
            vocabulary_.insert(word);
        }

        // 读取词频统计
        size_t wc_size;
        ifs.read(reinterpret_cast<char*>(&wc_size), sizeof(wc_size));
        for (size_t i = 0; i < wc_size; ++i) {
            size_t len;
            ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string word(len, ' ');
            ifs.read(&word[0], len);
            int count;
            ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
            word_count_[word] = count;
        }

        // 读取模型数据
        size_t model_size;
        ifs.read(reinterpret_cast<char*>(&model_size), sizeof(model_size));
        for (size_t i = 0; i < model_size; ++i) {
            int n_size;
            ifs.read(reinterpret_cast<char*>(&n_size), sizeof(n_size));

            size_t context_size;
            ifs.read(reinterpret_cast<char*>(&context_size), sizeof(context_size));

            auto& context_map = models_[n_size];
            for (size_t j = 0; j < context_size; ++j) {
                size_t ctx_len;
                ifs.read(reinterpret_cast<char*>(&ctx_len), sizeof(ctx_len));

                std::vector<std::string> context;
                context.reserve(ctx_len);
                for (size_t k = 0; k < ctx_len; ++k) {
                    size_t len;
                    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string word(len, ' ');
                    ifs.read(&word[0], len);
                    context.push_back(word);
                }

                size_t word_map_size;
                ifs.read(reinterpret_cast<char*>(&word_map_size), sizeof(word_map_size));

                auto& word_map = context_map[context];
                for (size_t k = 0; k < word_map_size; ++k) {
                    size_t len;
                    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
                    std::string word(len, ' ');
                    ifs.read(&word[0], len);
                    int count;
                    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
                    word_map[word] = count;
                }
            }
        }

        return true;
    } catch (...) {
        LOGE("Error loading model");
        return false;
    }
}

// TextPredictor实现
TextPredictor::TextPredictor(const std::string& model_path, int n,
                             const std::vector<std::string>* sample_texts)
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

void TextPredictor::add_to_history(const std::string& text) {
    user_history_.push_back(text);
    LOGD("Added to history. Current size: %zu/%d",
         user_history_.size(), HISTORY_THRESHOLD);

    if (user_history_.size() >= HISTORY_THRESHOLD) {
        LOGD("History threshold reached, training model...");
        force_training();
    }
}

std::vector<std::pair<std::string, double>> TextPredictor::predict(
        const std::string& context, int num_predictions) {

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
    for (const auto& text : user_history_) {
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
    ss << "n: " << model_->get_n() << "\n"
       << "Vocabulary size: " << model_->get_vocabulary_size() << "\n"
       << "Total words: " << model_->get_total_words() << "\n"
       << "History entries: " << user_history_.size() << "\n"
       << "Smoothing: " << model_->get_smoothing();
    return ss.str();
}
