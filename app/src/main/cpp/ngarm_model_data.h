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

// 模型参数封装结构体
struct NGramModelData {
    int n = 3;
    double smoothing = 0.1;
    int total_words = 0;
    std::unordered_set<std::string> vocabulary;
    std::unordered_map<std::string, int> word_count;
    std::unordered_map<int, std::unordered_map<std::vector<std::string>,
            std::unordered_map<std::string, int>, VectorHash>> models;
};
