#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>
#include <ostream>
#include <random>
#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

typedef std::unique_ptr<Ort::Env> OrtEnvPtr;
typedef std::unique_ptr<Ort::Session> OrtSessPtr;
typedef std::unique_ptr<Ort::RunOptions> OrtRunOpt;
int main(int argc, char* argv[])
{
    // generate input
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    uint feat_size = 80; // fixed for the sample model
    uint seq_size = 10; // arbitrary
    uint batch_size = 1; // arbitray
    uint sample_size = feat_size * seq_size * batch_size;
    std::vector<float> inputs(sample_size, 0);
    for (uint i = 0; i < sample_size; i++) {
        inputs[i] = distribution(generator);
        // std::cout << inputs[i] << std::endl;
    }

    auto ort_mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<int64_t> shape {batch_size, feat_size, seq_size};
    // Ort::Value in1 = Ort::Value::CreateTensor(ort_mem, inputs.data(), shape.size(), shape.data(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Ort::Value in1 = Ort::Value::CreateTensor(ort_mem, inputs.data(), sample_size * sizeof(float), shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    //
    // OrtEnvPtr ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_VERBOSE);
    OrtEnvPtr ort_env = OrtEnvPtr(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE));
    Ort::SessionOptions ort_opts;
    ort_opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_opts.SetExecutionMode(ORT_SEQUENTIAL);
    ort_opts.SetIntraOpNumThreads(1);
    ort_opts.SetInterOpNumThreads(1);

    // ort_opts.AddConfigEntry("session.load_model_format", "ORT");
    // ort_opts.AddConfigEntry("session.use_ort_model_bytes_directly", "1");

    // std::ifstream stream(argv[1], std::ios::in | std::ios::binary);
    // std::vector<uint8_t> model_bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());

    auto start = std::chrono::high_resolution_clock::now();
    OrtSessPtr sess = OrtSessPtr(new Ort::Session(*ort_env, argv[1], ort_opts));
    // OrtSessPtr sess = OrtSessPtr(new Ort::Session(*ort_env, model_bytes.data(), model_bytes.size(), ort_opts));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "time " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

    auto run_op = Ort::RunOptions();
    std::vector<const char*> input_names {"audio_signal"};
    std::vector<const char*> output_names {"logprobs"};
    auto run_rv = sess->Run(run_op, input_names.data(), &in1, 1, output_names.data(), 1);
    assert(run_rv.size() == 1);
    auto& out = run_rv[0];
    auto out_shape_type_info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = out_shape_type_info.GetShape();
    auto out_raw = out.GetTensorMutableData<float>();
    auto out_num_elem = out_shape_type_info.GetElementCount();

    // write ref
    // std::ofstream data_file;
    // data_file.open("ref.bin", std::ios::out | std::ios::binary);
    // data_file.write(reinterpret_cast<char*>(out_raw), out_num_elem * sizeof(float));
    // data_file.close();

    // read ref
    std::ifstream data_file;
    data_file.open("ref.bin", std::ios::in | std::ios::binary);
    auto read_size = sizeof(float) * out_num_elem;
    char * buf = new char[read_size];
    data_file.read(buf, read_size);

    // compare
    typedef Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MapCMatXf;
    MapCMatXf out_ei((float*)out_raw, out_shape[1], out_shape[2]);

    MapCMatXf ref_ei((float*)buf, out_shape[1], out_shape[2]);
    std::cout << "out shape is " << out_shape[1] << " " << out_shape[2] << std::endl;

    Eigen::Index maxRow, maxCol;
    float max = (out_ei - ref_ei).cwiseAbs().maxCoeff(&maxRow, &maxCol);
    std::cout << "largest diff is " << max << " at " << maxRow << " " << maxCol << std::endl;

    delete[] buf;
    return 0;
}

