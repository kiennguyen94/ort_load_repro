#include <iostream>
#include <memory>
#include <chrono>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

typedef std::unique_ptr<Ort::Env> OrtEnvPtr;
typedef std::unique_ptr<Ort::Session> OrtSessPtr;
typedef std::unique_ptr<Ort::RunOptions> OrtRunOpt;
int main(int argc, char* argv[])
{
    //
    // OrtEnvPtr ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE);
    OrtEnvPtr ort_env = OrtEnvPtr(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE));
    Ort::SessionOptions ort_opts;
    ort_opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_opts.SetExecutionMode(ORT_SEQUENTIAL);
    ort_opts.SetIntraOpNumThreads(1);
    ort_opts.SetInterOpNumThreads(1);
    auto start = std::chrono::high_resolution_clock::now();
    OrtSessPtr sess = OrtSessPtr(new Ort::Session(*ort_env, argv[1], ort_opts));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "time " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;
    return 0;
}

