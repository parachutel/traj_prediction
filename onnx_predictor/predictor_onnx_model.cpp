#include <vector>
#include <cmath>
#include <string>
#include <onnxruntime_cxx_api.h>

OnnxModel::OnnxModel() {}

template <typename T> 
T _vectorProduct(const std::vector<T>& v){
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

void OnnxModel::run(Ort::Session session,
                    Ort::AllocatorWithDefaultOptions allocator,
                    Ort::MemoryInfo memoryInfo,
                    const std::vector<std::vector<float>>& inputTensorValues,
                    std::vector<std::vector<float>>& results) {

    std::vector<const char*> inputNames{
        session.GetInputName(0, allocator),
        session.GetInputName(1, allocator), 
        session.GetInputName(2, allocator)
    };
    std::vector<const char*> outputNames{
        session.GetOutputName(0, allocator)
    };

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    for (int i = 0; i < 3; ++i){
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        size_t inputTensorSize = _vectorProduct(inputDims);
        inputTensors.push_back(
            Ort::Value::CreateTensor<float>(
                memoryInfo, inputTensorValues[i].data(), inputTensorSize, 
                inputDims.data(), inputDims.size()
            )
        );
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    size_t outputTensorSize = _vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);
    outputTensors.push_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()
        )
    );

    // outputTensors are modified in-place
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 3, outputNames.data(),
                outputTensors.data(), 1);

    results;
}



Ort::Session OnnxModel::createSession(const std::string model_file_path) {
    Ort::Env env;
    Ort::Session session(env, model_file_path.c_str(), 
                         Ort::SessionOptions{nullptr});
    return session;
}


Ort::AllocatorWithDefaultOptions OnnxModel::allocateMemory() {
    Ort::AllocatorWithDefaultOptions allocator;
    return allocator;
}


Ort::MemoryInfo OnnxModel::getMemoryInfo() {
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return memoryInfo;
}