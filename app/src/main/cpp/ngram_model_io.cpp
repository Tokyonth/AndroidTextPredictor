#include "ngarm_model_data.h"
#include "jni_log.h"
#include <cstdio>
#include <vector>
#include <memory>
#include <type_traits>

const size_t IO_BUFFER_SIZE = 64 * 1024;

bool save_model_data(NGramModelData &data, const std::string &file_path) {
    // 检查total_words是否有效
    if (data.total_words <= 0) {
        LOGE("Invalid total_words value: %d (must be positive)", data.total_words);
        return false;
    }

    FILE *fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        LOGE("Failed to open file for saving: %s", file_path.c_str());
        return false;
    }

    // 使用智能指针管理缓冲区
    std::unique_ptr<char[]> buf(new char[IO_BUFFER_SIZE]);
    setvbuf(fp, buf.get(), _IOFBF, IO_BUFFER_SIZE);

    try {
        // 逐个写入基本参数（修复核心：避免批量读写）
        // 写入n
        if (fwrite(&data.n, sizeof(data.n), 1, fp) != 1) {
            LOGE("Failed to write n");
            fclose(fp);
            return false;
        }

        // 写入smoothing
        if (fwrite(&data.smoothing, sizeof(data.smoothing), 1, fp) != 1) {
            LOGE("Failed to write smoothing");
            fclose(fp);
            return false;
        }

        // 写入total_words（关键修复）
        if (fwrite(&data.total_words, sizeof(data.total_words), 1, fp) != 1) {
            LOGE("Failed to write total_words");
            fclose(fp);
            return false;
        }

        // 写入词频统计
        size_t wc_size = data.word_count.size();
        if (fwrite(&wc_size, sizeof(wc_size), 1, fp) != 1) {
            LOGE("Failed to write wc_size");
            fclose(fp);
            return false;
        }

        std::string word_buf;
        for (const auto &entry: data.word_count) {
            const std::string &word = entry.first;
            size_t len = word.size();

            if (fwrite(&len, sizeof(len), 1, fp) != 1) {
                LOGE("Failed to write word length");
                fclose(fp);
                return false;
            }

            if (fwrite(word.data(), len, 1, fp) != 1) {
                LOGE("Failed to write word data");
                fclose(fp);
                return false;
            }

            if (fwrite(&entry.second, sizeof(entry.second), 1, fp) != 1) {
                LOGE("Failed to write word count");
                fclose(fp);
                return false;
            }
        }

        // 写入模型数据
        using ModelMapType = std::decay_t<decltype(data.models)>;
        std::vector<std::pair<int, ModelMapType::mapped_type>> model_vec(
                data.models.begin(), data.models.end()
        );

        size_t model_size = model_vec.size();
        if (fwrite(&model_size, sizeof(model_size), 1, fp) != 1) {
            LOGE("Failed to write model_size");
            fclose(fp);
            return false;
        }

        for (const auto &model_entry: model_vec) {
            int n_size = model_entry.first;
            if (fwrite(&n_size, sizeof(n_size), 1, fp) != 1) {
                LOGE("Failed to write n_size");
                fclose(fp);
                return false;
            }

            const auto &context_map = model_entry.second;
            using ContextMapType = std::decay_t<decltype(context_map)>;
            std::vector<std::pair<std::vector<std::string>, ContextMapType::mapped_type>> context_vec(
                    context_map.begin(), context_map.end()
            );

            size_t context_size = context_vec.size();
            if (fwrite(&context_size, sizeof(context_size), 1, fp) != 1) {
                LOGE("Failed to write context_size");
                fclose(fp);
                return false;
            }

            for (const auto &context_entry: context_vec) {
                const std::vector<std::string> &context = context_entry.first;
                size_t ctx_len = context.size();

                if (fwrite(&ctx_len, sizeof(ctx_len), 1, fp) != 1) {
                    LOGE("Failed to write ctx_len");
                    fclose(fp);
                    return false;
                }

                for (const auto &word: context) {
                    size_t len = word.size();
                    if (fwrite(&len, sizeof(len), 1, fp) != 1) {
                        LOGE("Failed to write context word length");
                        fclose(fp);
                        return false;
                    }

                    if (fwrite(word.data(), len, 1, fp) != 1) {
                        LOGE("Failed to write context word data");
                        fclose(fp);
                        return false;
                    }
                }

                const auto &word_map = context_entry.second;
                std::vector<std::pair<std::string, int>> word_vec(
                        word_map.begin(), word_map.end()
                );

                size_t word_map_size = word_vec.size();
                if (fwrite(&word_map_size, sizeof(word_map_size), 1, fp) != 1) {
                    LOGE("Failed to write word_map_size");
                    fclose(fp);
                    return false;
                }

                for (const auto &word_entry: word_vec) {
                    size_t len = word_entry.first.size();
                    if (fwrite(&len, sizeof(len), 1, fp) != 1) {
                        LOGE("Failed to write entry word length");
                        fclose(fp);
                        return false;
                    }

                    if (fwrite(word_entry.first.data(), len, 1, fp) != 1) {
                        LOGE("Failed to write entry word data");
                        fclose(fp);
                        return false;
                    }

                    if (fwrite(&word_entry.second, sizeof(word_entry.second), 1, fp) != 1) {
                        LOGE("Failed to write entry word count");
                        fclose(fp);
                        return false;
                    }
                }
            }
        }

        fclose(fp);
        LOGD("Model saved successfully, total_words: %d", data.total_words);
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

    // 使用智能指针管理缓冲区
    std::unique_ptr<char[]> buf(new char[IO_BUFFER_SIZE]);
    setvbuf(fp, buf.get(), _IOFBF, IO_BUFFER_SIZE);

    try {
        // 清空现有数据
        data.models.clear();
        data.word_count.clear();
        data.vocabulary.clear();
        data.total_words = 0;  // 初始化为0，便于检测是否读取成功

        // 逐个读取基本参数（修复核心）
        // 读取n
        if (fread(&data.n, sizeof(data.n), 1, fp) != 1) {
            LOGE("Failed to read n");
            fclose(fp);
            return false;
        }

        // 读取smoothing
        if (fread(&data.smoothing, sizeof(data.smoothing), 1, fp) != 1) {
            LOGE("Failed to read smoothing");
            fclose(fp);
            return false;
        }

        // 读取total_words（关键修复）
        if (fread(&data.total_words, sizeof(data.total_words), 1, fp) != 1) {
            LOGE("Failed to read total_words");
            fclose(fp);
            return false;
        }

        // 验证total_words是否合理
        if (data.total_words <= 0) {
            LOGE("Loaded invalid total_words: %d (may indicate corrupt file)", data.total_words);
            fclose(fp);
            return false;
        }

        // 读取词频统计
        size_t wc_size;
        if (fread(&wc_size, sizeof(wc_size), 1, fp) != 1) {
            LOGE("Failed to read wc_size");
            fclose(fp);
            return false;
        }

        data.word_count.reserve(wc_size);
        std::string word;
        int count;
        for (size_t i = 0; i < wc_size; ++i) {
            size_t len;
            if (fread(&len, sizeof(len), 1, fp) != 1) {
                LOGE("Failed to read word length");
                fclose(fp);
                return false;
            }

            word.resize(len);
            if (fread(&word[0], len, 1, fp) != 1) {
                LOGE("Failed to read word data");
                fclose(fp);
                return false;
            }

            if (fread(&count, sizeof(count), 1, fp) != 1) {
                LOGE("Failed to read word count");
                fclose(fp);
                return false;
            }

            data.word_count[word] = count;
        }

        // 重建vocabulary
        data.vocabulary.reserve(data.word_count.size());
        for (const auto &entry: data.word_count) {
            data.vocabulary.insert(entry.first);
        }

        // 读取模型数据
        size_t model_size;
        if (fread(&model_size, sizeof(model_size), 1, fp) != 1) {
            LOGE("Failed to read model_size");
            fclose(fp);
            return false;
        }

        data.models.reserve(model_size);
        for (size_t i = 0; i < model_size; ++i) {
            int n_size;
            if (fread(&n_size, sizeof(n_size), 1, fp) != 1) {
                LOGE("Failed to read n_size");
                fclose(fp);
                return false;
            }

            size_t context_size;
            if (fread(&context_size, sizeof(context_size), 1, fp) != 1) {
                LOGE("Failed to read context_size");
                fclose(fp);
                return false;
            }

            auto &context_map = data.models[n_size];
            context_map.reserve(context_size);

            for (size_t j = 0; j < context_size; ++j) {
                size_t ctx_len;
                if (fread(&ctx_len, sizeof(ctx_len), 1, fp) != 1) {
                    LOGE("Failed to read ctx_len");
                    fclose(fp);
                    return false;
                }

                std::vector<std::string> context;
                context.reserve(ctx_len);
                for (size_t k = 0; k < ctx_len; ++k) {
                    size_t len;
                    if (fread(&len, sizeof(len), 1, fp) != 1) {
                        LOGE("Failed to read context word length");
                        fclose(fp);
                        return false;
                    }

                    word.resize(len);
                    if (fread(&word[0], len, 1, fp) != 1) {
                        LOGE("Failed to read context word data");
                        fclose(fp);
                        return false;
                    }

                    context.push_back(word);
                }

                size_t word_map_size;
                if (fread(&word_map_size, sizeof(word_map_size), 1, fp) != 1) {
                    LOGE("Failed to read word_map_size");
                    fclose(fp);
                    return false;
                }

                auto &word_map = context_map[context];
                word_map.reserve(word_map_size);

                for (size_t k = 0; k < word_map_size; ++k) {
                    size_t len;
                    if (fread(&len, sizeof(len), 1, fp) != 1) {
                        LOGE("Failed to read entry word length");
                        fclose(fp);
                        return false;
                    }

                    word.resize(len);
                    if (fread(&word[0], len, 1, fp) != 1) {
                        LOGE("Failed to read entry word data");
                        fclose(fp);
                        return false;
                    }

                    if (fread(&count, sizeof(count), 1, fp) != 1) {
                        LOGE("Failed to read entry word count");
                        fclose(fp);
                        return false;
                    }

                    word_map[word] = count;
                }
            }
        }

        fclose(fp);
        LOGD("Model loaded successfully, total_words: %d", data.total_words);
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
