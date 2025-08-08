#include "ngarm_model_data.h"
#include "jni_log.h"
#include <cstdio>
#include <vector>
#include <memory>  // 引入智能指针

const size_t IO_BUFFER_SIZE = 64 * 1024;

bool save_model_data(NGramModelData &data, const std::string &file_path) {
    FILE *fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        LOGE("Failed to open file for saving: %s", file_path.c_str());
        return false;
    }

    // 使用unique_ptr自动管理缓冲区内存，离开作用域时自动释放
    std::unique_ptr<char[]> buf(new char[IO_BUFFER_SIZE]);
    setvbuf(fp, buf.get(), _IOFBF, IO_BUFFER_SIZE);  // 传入原始指针

    try {
        // 写入基本参数
        const size_t basic_data_size =
                sizeof(data.n) + sizeof(data.smoothing) + sizeof(data.total_words);
        fwrite(&data.n, basic_data_size, 1, fp);

        // 写入词频统计（复用字符串变量）
        size_t wc_size = data.word_count.size();
        fwrite(&wc_size, sizeof(wc_size), 1, fp);
        std::string word_buf;
        for (const auto &entry: data.word_count) {
            const std::string &word = entry.first;
            size_t len = word.size();
            fwrite(&len, sizeof(len), 1, fp);
            fwrite(word.data(), len, 1, fp);
            fwrite(&entry.second, sizeof(entry.second), 1, fp);
        }

        // 写入模型数据（转为vector顺序遍历）
        std::vector<std::pair<int, decltype(data.models)::mapped_type>> model_vec(
                data.models.begin(), data.models.end()
        );
        size_t model_size = model_vec.size();
        fwrite(&model_size, sizeof(model_size), 1, fp);

        for (const auto &model_entry: model_vec) {
            int n_size = model_entry.first;
            fwrite(&n_size, sizeof(n_size), 1, fp);

            const auto &context_map = model_entry.second;
            // 使用std::decay_t移除引用和const限定，获取原始容器类型
            using ContextMapType = std::decay_t<decltype(context_map)>;
            std::vector<std::pair<std::vector<std::string>, ContextMapType::mapped_type>> context_vec(
                    context_map.begin(), context_map.end()
            );
            size_t context_size = context_vec.size();
            fwrite(&context_size, sizeof(context_size), 1, fp);

            for (const auto &context_entry: context_vec) {
                const std::vector<std::string> &context = context_entry.first;
                size_t ctx_len = context.size();
                fwrite(&ctx_len, sizeof(ctx_len), 1, fp);

                for (const auto &word: context) {
                    size_t len = word.size();
                    fwrite(&len, sizeof(len), 1, fp);
                    fwrite(word.data(), len, 1, fp);
                }

                const auto &word_map = context_entry.second;
                std::vector<std::pair<std::string, int>> word_vec(
                        word_map.begin(), word_map.end()
                );
                size_t word_map_size = word_vec.size();
                fwrite(&word_map_size, sizeof(word_map_size), 1, fp);

                for (const auto &word_entry: word_vec) {
                    size_t len = word_entry.first.size();
                    fwrite(&len, sizeof(len), 1, fp);
                    fwrite(word_entry.first.data(), len, 1, fp);
                    fwrite(&word_entry.second, sizeof(word_entry.second), 1, fp);
                }
            }
        }

        fclose(fp);
        return true;
    } catch (const std::exception &e) {
        LOGE("Error saving model: %s", e.what());
        fclose(fp);
        return false;
    } catch (...) {
        LOGE("Unknown error saving model");
        fclose(fp);
        return false;
    }
}

bool load_model_data(NGramModelData &data, const std::string &file_path) {
    FILE *fp = fopen(file_path.c_str(), "rb");
    if (!fp) {
        LOGE("Failed to open file for loading: %s", file_path.c_str());
        return false;
    }

    // 使用unique_ptr自动管理缓冲区内存
    std::unique_ptr<char[]> buf(new char[IO_BUFFER_SIZE]);
    setvbuf(fp, buf.get(), _IOFBF, IO_BUFFER_SIZE);

    try {
        // 清空现有数据
        data.models.clear();
        data.word_count.clear();
        data.vocabulary.clear();
        data.total_words = 0;

        // 读取基本参数
        const size_t basic_data_size =
                sizeof(data.n) + sizeof(data.smoothing) + sizeof(data.total_words);
        fread(&data.n, basic_data_size, 1, fp);

        // 读取词频统计
        size_t wc_size;
        fread(&wc_size, sizeof(wc_size), 1, fp);
        data.word_count.reserve(wc_size);
        std::string word;
        int count;
        for (size_t i = 0; i < wc_size; ++i) {
            size_t len;
            fread(&len, sizeof(len), 1, fp);
            word.resize(len);
            fread(&word[0], len, 1, fp);
            fread(&count, sizeof(count), 1, fp);
            data.word_count[word] = count;
        }

        // 重建vocabulary
        data.vocabulary.reserve(data.word_count.size());
        for (const auto &entry: data.word_count) {
            data.vocabulary.insert(entry.first);
        }

        // 读取模型数据
        size_t model_size;
        fread(&model_size, sizeof(model_size), 1, fp);
        data.models.reserve(model_size);

        for (size_t i = 0; i < model_size; ++i) {
            int n_size;
            fread(&n_size, sizeof(n_size), 1, fp);

            size_t context_size;
            fread(&context_size, sizeof(context_size), 1, fp);

            auto &context_map = data.models[n_size];
            context_map.reserve(context_size);

            for (size_t j = 0; j < context_size; ++j) {
                size_t ctx_len;
                fread(&ctx_len, sizeof(ctx_len), 1, fp);

                std::vector<std::string> context;
                context.reserve(ctx_len);
                for (size_t k = 0; k < ctx_len; ++k) {
                    size_t len;
                    fread(&len, sizeof(len), 1, fp);
                    word.resize(len);
                    fread(&word[0], len, 1, fp);
                    context.push_back(word);
                }

                size_t word_map_size;
                fread(&word_map_size, sizeof(word_map_size), 1, fp);

                auto &word_map = context_map[context];
                word_map.reserve(word_map_size);

                for (size_t k = 0; k < word_map_size; ++k) {
                    size_t len;
                    fread(&len, sizeof(len), 1, fp);
                    word.resize(len);
                    fread(&word[0], len, 1, fp);
                    fread(&count, sizeof(count), 1, fp);
                    word_map[word] = count;
                }
            }
        }

        fclose(fp);
        return true;
    } catch (const std::exception &e) {
        LOGE("Error loading model: %s", e.what());
        fclose(fp);
        return false;
    } catch (...) {
        LOGE("Unknown error loading model");
        fclose(fp);
        return false;
    }

}
