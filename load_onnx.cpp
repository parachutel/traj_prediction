#include <onnxruntime_cxx_api.h>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <typename T> 
T vectorProduct(const std::vector<T>& v){
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T> 
void printVector(const std::vector<T>& v){
    std::cout << "(";
    for (int i = 0; i < v.size(); i++){
        if (i < v.size() - 1){
            std::cout << v[i] << ", ";
        }
        else{
            std::cout << v[i] << ")" << std::endl;
        }
    }
}

int main(int argc, char* argv[]){
    std::string modelFilepath{"/home/shengli/traj_prediction/save/onnx/"
        "cvae_maskedEHE=True_bs=512_input=2_pred=6_stride=2_seed=0_run-01/"
        "predictor.onnx"};
    Ort::Env env;
    Ort::Session session(env, modelFilepath.c_str(), Ort::SessionOptions{nullptr});

    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> inputNames{session.GetInputName(0, allocator),
        session.GetInputName(1, allocator), session.GetInputName(2, allocator)};
    std::vector<const char*> outputNames{session.GetOutputName(0, allocator)};

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);


    for (int i = 0; i < 3; ++i){
        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        size_t inputTensorSize = vectorProduct(inputDims);
        std::cout << inputNames[i] << " shape = ";
        printVector(inputDims);
        std::vector<float> inputTensorValues(inputTensorSize);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
            inputDims.size())
        );
    }

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    size_t outputTensorSize = vectorProduct(outputDims);
    std::cout << outputNames[0] <<  " shape = ";
    printVector(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size())
    );

    // outputTensors are modified in-place
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 3, outputNames.data(),
                outputTensors.data(), 1);

    
    return 1;
}
