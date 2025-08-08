#include <jni.h>
#include <string>
#include <vector>

#include "ngram_model.h"
#include "jni_log.h"

// 存储TextPredictor实例的映射
static std::unordered_map<jlong, std::unique_ptr<TextPredictor>> predictors;
static jlong next_predictor_id = 1;

extern "C" JNIEXPORT jlong JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_createPredictor(
        JNIEnv *env, jobject thiz, jstring model_path, jint n, jobjectArray sample_texts) {
    (void) thiz;

    const char *path = env->GetStringUTFChars(model_path, nullptr);
    if (!path) return 0;

    std::vector<std::string> samples;
    if (sample_texts) {
        jsize len = env->GetArrayLength(sample_texts);
        for (jsize i = 0; i < len; ++i) {
            auto text = (jstring) env->GetObjectArrayElement(sample_texts, i);
            const char *ctext = env->GetStringUTFChars(text, nullptr);
            if (ctext) {
                samples.emplace_back(ctext);
                env->ReleaseStringUTFChars(text, ctext);
            }
            env->DeleteLocalRef(text);
        }
    }

    jlong id = next_predictor_id++;
    predictors[id] = std::make_unique<TextPredictor>(
            std::string(path), n, samples.empty() ? nullptr : &samples);

    env->ReleaseStringUTFChars(model_path, path);
    return id;
}

extern "C" JNIEXPORT void JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_addToHistory(
        JNIEnv *env, jobject thiz, jlong predictor_id, jstring text) {
    (void) thiz;

    auto it = predictors.find(predictor_id);
    if (it == predictors.end()) return;

    const char *ctext = env->GetStringUTFChars(text, nullptr);
    if (ctext) {
        it->second->add_to_history(std::string(ctext));
        env->ReleaseStringUTFChars(text, ctext);
    }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_predict(
        JNIEnv *env, jobject thiz, jlong predictor_id, jstring context, jint num_predictions) {
    (void) thiz;

    auto it = predictors.find(predictor_id);
    if (it == predictors.end()) return nullptr;

    const char *ccontext = env->GetStringUTFChars(context, nullptr);
    if (!ccontext) return nullptr;

    auto results = it->second->predict(std::string(ccontext), num_predictions);
    env->ReleaseStringUTFChars(context, ccontext);

    // 创建结果数组
    jclass pair_class = env->FindClass("android/util/Pair");
    jmethodID pair_constructor = env->GetMethodID(pair_class, "<init>",
                                                  "(Ljava/lang/Object;Ljava/lang/Object;)V");

    jobjectArray result_array = env->NewObjectArray(results.size(), pair_class, nullptr);

    for (size_t i = 0; i < results.size(); ++i) {
        jstring word = env->NewStringUTF(results[i].first.c_str());
        jdouble prob = results[i].second;
        jobject prob_obj = env->NewObject(env->FindClass("java/lang/Double"),
                                          env->GetMethodID(env->FindClass("java/lang/Double"),
                                                           "<init>", "(D)V"),
                                          prob);

        jobject pair = env->NewObject(pair_class, pair_constructor, word, prob_obj);
        env->SetObjectArrayElement(result_array, i, pair);

        env->DeleteLocalRef(word);
        env->DeleteLocalRef(prob_obj);
        env->DeleteLocalRef(pair);
    }

    env->DeleteLocalRef(pair_class);
    return result_array;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_forceTraining(
        JNIEnv *env, jobject thiz, jlong predictor_id) {
    (void) env;
    (void) thiz;

    auto it = predictors.find(predictor_id);
    if (it == predictors.end()) return JNI_FALSE;

    return it->second->force_training() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_clearHistory(
        JNIEnv *env, jobject thiz, jlong predictor_id) {
    (void) env;
    (void) thiz;

    auto it = predictors.find(predictor_id);
    if (it != predictors.end()) {
        it->second->clear_history();
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_getModelInfo(
        JNIEnv *env, jobject thiz, jlong predictor_id) {
    (void) thiz;

    auto it = predictors.find(predictor_id);
    if (it == predictors.end()) {
        return env->NewStringUTF("No model available");
    }

    std::string info = it->second->get_model_info();
    return env->NewStringUTF(info.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_tokyonth_textpredictor_TextPredictorNative_destroyPredictor(
        JNIEnv *env, jobject thiz, jlong predictor_id) {
    (void) env;
    (void) thiz;

    LOGD("Destroying predictor: %ld", predictor_id);

    predictors.erase(predictor_id);
}
