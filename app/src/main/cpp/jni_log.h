#ifndef JNI_LOG_H
#define JNI_LOG_H

#include <android/log.h>
#include <pthread.h>
#include <string>

// 日志标签，可根据项目修改
#ifndef LOG_TAG
#define LOG_TAG "NgramNative"
#endif

// 日志级别定义
enum LogLevel {
    LOG_VERBOSE = ANDROID_LOG_VERBOSE,
    LOG_DEBUG = ANDROID_LOG_DEBUG,
    LOG_INFO = ANDROID_LOG_INFO,
    LOG_WARN = ANDROID_LOG_WARN,
    LOG_ERROR = ANDROID_LOG_ERROR,
    LOG_FATAL = ANDROID_LOG_FATAL
};

class JniLog {
public:
    static void isEnableLogging(bool isEnable) { s_isEnable = isEnable; }

    // 配置日志是否显示线程ID
    static void setShowThreadId(bool show) { s_showThreadId = show; }

    // 配置日志是否显示文件和行号
    static void setShowFileLine(bool show) { s_showFileLine = show; }

    // 核心日志函数
    static void log(LogLevel level, const char *tag, const char *file, int line,
                    const char *format, ...);

private:
    static bool s_isEnable;
    static bool s_showThreadId;  // 是否显示线程ID
    static bool s_showFileLine;  // 是否显示文件和行号

    // 获取当前线程ID字符串
    static std::string getThreadId();

    // 格式化日志前缀（线程ID、文件行号）
    static std::string formatPrefix(const char *file, int line);
};

// 日志宏定义 - 自动包含文件名和行号，使用默认标签
#define LOGV(...) JniLog::log(LOG_VERBOSE, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGD(...) JniLog::log(LOG_DEBUG, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGI(...) JniLog::log(LOG_INFO, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGW(...) JniLog::log(LOG_WARN, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGE(...) JniLog::log(LOG_ERROR, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGF(...) JniLog::log(LOG_FATAL, LOG_TAG, __FILE__, __LINE__, __VA_ARGS__)

// 日志宏定义 - 支持自定义标签
#define LOGV_TAG(tag, ...) JniLog::log(LOG_VERBOSE, tag, __FILE__, __LINE__, __VA_ARGS__)
#define LOGD_TAG(tag, ...) JniLog::log(LOG_DEBUG, tag, __FILE__, __LINE__, __VA_ARGS__)
#define LOGI_TAG(tag, ...) JniLog::log(LOG_INFO, tag, __FILE__, __LINE__, __VA_ARGS__)
#define LOGW_TAG(tag, ...) JniLog::log(LOG_WARN, tag, __FILE__, __LINE__, __VA_ARGS__)
#define LOGE_TAG(tag, ...) JniLog::log(LOG_ERROR, tag, __FILE__, __LINE__, __VA_ARGS__)
#define LOGF_TAG(tag, ...) JniLog::log(LOG_FATAL, tag, __FILE__, __LINE__, __VA_ARGS__)

#endif // JNI_LOG_H
