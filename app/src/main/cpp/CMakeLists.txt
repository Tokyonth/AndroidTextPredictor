# 指定CMake最低版本要求
cmake_minimum_required(VERSION 3.18.1)

# 项目名称
project("predictor")

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 定义源文件列表
set(SOURCE_FILES
        native-lib.cpp
        ngram_model.cpp
        jni_log.cpp
        ngram_model_io.cpp
)

# 定义头文件目录
include_directories(.)

# 添加编译选项以优化性能
add_compile_options(
        -O3
        -ffast-math
        -funroll-loops
        -ftree-vectorize
        -Wall
        -Wextra
        #-Werror
)

# 针对不同架构的额外优化
if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_compile_options(-march=armv8-a+simd)
elseif (${ANDROID_ABI} STREQUAL "armeabi-v7a")
    add_compile_options(-march=armv7-a -mfpu=neon)
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    add_compile_options(-march=x86-64 -msse4.2 -mpopcnt)
endif ()

# 创建共享库
add_library(
        # 库名称，在Java中加载时使用
        predictor

        # 库类型：共享库
        SHARED

        # 源文件
        ${SOURCE_FILES}
)

# 查找Android日志库
find_library(
        log-lib
        log
)

# 链接库
target_link_libraries(
        predictor
        ${log-lib}
)
