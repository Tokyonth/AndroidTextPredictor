#include "jni_log.h"
#include <cstdarg>
#include <cstring>
#include <sstream>

// 静态成员初始化
bool JniLog::s_showThreadId = true;
bool JniLog::s_showFileLine = true;

std::string JniLog::getThreadId() {
    std::stringstream ss;
    ss << pthread_self();  // 获取当前线程ID
    return ss.str();
}

std::string JniLog::formatPrefix(const char* file, int line) {
    std::string prefix;

    // 添加线程ID
    if (s_showThreadId) {
        prefix += "[" + getThreadId() + "] ";
    }

    // 添加文件名和行号
    if (s_showFileLine) {
        // 提取文件名（去掉路径，只保留文件名）
        const char* fileName = strrchr(file, '/');
        if (fileName) {
            fileName++;  // 跳过 '/'
        } else {
            fileName = file;  // 如果没有路径分隔符，直接使用文件名
        }

        // 拼接文件名和行号
        std::stringstream ss;
        ss << "[" << fileName << ":" << line << "] ";
        prefix += ss.str();
    }

    return prefix;
}

void JniLog::log(LogLevel level, const char* tag, const char* file, int line,
                 const char* format, ...) {
    if (!tag || !file || !format) {
        return;
    }

    // 格式化日志前缀（线程ID、文件行号）
    std::string prefix = formatPrefix(file, line);

    // 处理可变参数
    va_list args;
    va_start(args, format);

    // 计算日志内容长度
    int contentLength = vsnprintf(nullptr, 0, format, args) + 1;  // +1 包含终止符
    va_end(args);  // 重置可变参数列表

    // 分配缓冲区：前缀长度 + 日志内容长度
    int totalLength = prefix.length() + contentLength;
    char* logBuffer = new char[totalLength];
    if (!logBuffer) {
        return;
    }

    // 复制前缀到缓冲区
    strcpy(logBuffer, prefix.c_str());

    // 格式化日志内容到缓冲区
    va_start(args, format);
    vsnprintf(logBuffer + prefix.length(), contentLength, format, args);
    va_end(args);

    // 输出到Android日志系统
    __android_log_write(level, tag, logBuffer);

    // 释放缓冲区
    delete[] logBuffer;
}
